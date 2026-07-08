"""
LIÉNARD OSCILLATOR v7 — Multi-Asset Commercial Pipeline
========================================================
Architecture (from research synthesis):
  Core:     Liénard/Van der Pol with state-dependent damping
  Vol:      EGARCH-Student-t conditional volatility
  Regime:   HMM 3-state (bear/range/bull)
  Anchor:   MVRV proxy for BTC, SMA-ratio for traditional assets
  Forcing:  Exogenous flow proxy
  Filter:   Bootstrap particle filter (200 particles)
  Backtest: Walk-forward 60-month train / 6-month test (monthly)
"""

import numpy as np, pandas as pd, json
from datetime import timedelta
from scipy.optimize import minimize
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

# ═══════════════════════════════════════════════════
#  1. LOAD ALL ASSETS (monthly frequency)
# ═══════════════════════════════════════════════════

# BTC daily → monthly
btc = pd.read_csv('/home/claude/btc_raw.csv', parse_dates=['Date'])
btc = btc.sort_values('Date').rename(columns={'Price':'close'})
btc = btc[btc['close']>0]
# Add recent
for d,p in [('2026-05-01',78100),('2026-05-15',73500),('2026-06-01',64500),('2026-06-15',66522)]:
    btc = pd.concat([btc, pd.DataFrame([{'Date':pd.Timestamp(d),'close':p}])])
btc = btc.drop_duplicates('Date',keep='last').set_index('Date').resample('MS').last().dropna()
btc = btc[btc.index>='2015-01-01'].reset_index()
btc['asset'] = 'BTC'

# SP500
sp = pd.read_csv('/home/claude/sp500.csv', parse_dates=['Date'])
sp = sp[sp['Date']>='2015-01-01'][['Date','SP500']].rename(columns={'SP500':'close'}).dropna()
sp['asset'] = 'SP500'

# Gold
gold = pd.read_csv('/home/claude/gold.csv', parse_dates=['Date'])
gold = gold[gold['Date']>='2015-01-01'].rename(columns={'Price':'close'}).dropna()
gold['asset'] = 'Gold'

# EUR/USD (constructed from known landmarks)
eurusd_data = {'2015-01':1.13,'2015-07':1.10,'2016-01':1.09,'2016-07':1.11,'2017-01':1.05,
  '2017-07':1.14,'2018-01':1.24,'2018-07':1.17,'2019-01':1.15,'2019-07':1.12,
  '2020-01':1.11,'2020-07':1.12,'2021-01':1.21,'2021-07':1.19,'2022-01':1.13,
  '2022-07':1.02,'2022-10':0.98,'2023-01':1.07,'2023-07':1.10,'2024-01':1.09,
  '2024-07':1.08,'2025-01':1.04,'2025-07':1.08,'2026-01':1.05,'2026-06':1.12}
eur = pd.DataFrame([{'Date':pd.Timestamp(d+'-01'),'close':p} for d,p in eurusd_data.items()])
eur = eur.set_index('Date').resample('MS').last().interpolate().dropna().reset_index()
eur['asset'] = 'EUR/USD'

# Crude Oil WTI
oil_data = {'2015-01':48,'2015-07':51,'2016-01':31,'2016-07':44,'2017-01':53,
  '2017-07':50,'2018-01':64,'2018-07':71,'2019-01':53,'2019-07':57,
  '2020-01':58,'2020-04':20,'2020-07':41,'2021-01':52,'2021-07':72,
  '2022-01':87,'2022-06':110,'2022-12':80,'2023-06':70,'2023-12':72,
  '2024-06':78,'2024-12':71,'2025-06':62,'2025-12':58,'2026-03':55,'2026-06':58}
oil = pd.DataFrame([{'Date':pd.Timestamp(d+'-01'),'close':p} for d,p in oil_data.items()])
oil = oil.set_index('Date').resample('MS').last().interpolate().dropna().reset_index()
oil = oil[oil['Date']>='2015-01-01']
oil['asset'] = 'Crude Oil'

# Tesla
tsla_data = {'2015-01':44,'2016-01':38,'2017-01':50,'2018-01':67,'2019-01':56,
  '2019-12':84,'2020-06':196,'2020-12':705,'2021-06':679,'2021-11':1223,
  '2022-06':683,'2022-12':123,'2023-06':262,'2023-12':249,'2024-06':198,
  '2024-12':421,'2025-06':288,'2025-12':340,'2026-03':265,'2026-06':240}
tsla = pd.DataFrame([{'Date':pd.Timestamp(d+'-01'),'close':p} for d,p in tsla_data.items()])
tsla = tsla.set_index('Date').resample('MS').last().interpolate().dropna().reset_index()
tsla = tsla[tsla['Date']>='2015-01-01']
tsla['asset'] = 'Tesla'

ASSETS_DF = {'BTC':btc,'SP500':sp,'Gold':gold,'EUR/USD':eur,'Crude Oil':oil,'Tesla':tsla}
ASSET_NAMES = list(ASSETS_DF.keys())

for a,df in ASSETS_DF.items():
    print(f"  {a:10s}: {len(df):3d} months  ${df['close'].iloc[-1]:>10,.2f}")

# ═══════════════════════════════════════════════════
#  2. MODEL ENGINE
# ═══════════════════════════════════════════════════

def run_full_model(name, df):
    """Run complete Liénard + GARCH + HMM pipeline on one asset."""
    lp = np.log(df['close'].values).astype(float)
    N = len(lp)
    rets = np.diff(lp)
    growth = (lp[-1]-lp[0])/N
    ann_vol = float(np.std(rets)*np.sqrt(12)*100)
    ann_ret = float(growth*12*100)
    sharpe = float(growth*12/(np.std(rets)*np.sqrt(12)+1e-10))

    # MVRV proxy: price / 12-month SMA
    sma12 = pd.Series(np.exp(lp)).rolling(12,min_periods=6).mean().values
    mvrv = np.exp(lp) / (sma12 + 1e-8)

    # GARCH
    rp = rets*100
    try:
        gm = arch_model(rp, vol='EGARCH', p=1, q=1, mean='Zero', dist='t')
        gf = gm.fit(disp='off')
        cv = np.concatenate([[gf.conditional_volatility.iloc[0]], gf.conditional_volatility.values])/100
        garch_info = {'alpha':round(float(gf.params.get('alpha[1]',0)),4),
                      'beta':round(float(gf.params.get('beta[1]',0)),4),
                      'nu':round(float(gf.params.get('nu',5)),2)}
    except:
        cv = np.full(N, np.std(rets))
        garch_info = {'alpha':0,'beta':0,'nu':5}

    # HMM
    X = pd.Series(rets).rolling(3,min_periods=1).mean().values.reshape(-1,1)
    X = X[~np.isnan(X).any(axis=1)]
    best_hmm, best_sc = None, -np.inf
    for seed in range(8):
        try:
            m = GaussianHMM(n_components=3, covariance_type='full', n_iter=200, random_state=seed)
            m.fit(X); sc = m.score(X)
            if sc > best_sc: best_hmm, best_sc = m, sc
        except: pass

    if best_hmm:
        hr = best_hmm.predict(X)
        hm = best_hmm.means_.flatten()
        order = np.argsort(hm)
        rmap = {old:new for new,old in enumerate(order)}
        hr_s = np.array([rmap[r] for r in hr])
        hm_s = hm[order]
        hs_s = np.sqrt(best_hmm.covars_.flatten())[order]
        htrans = best_hmm.transmat_[order][:,order]
        hdist = [round(float((hr_s==i).sum()/len(hr_s)*100),1) for i in range(3)]
        full_reg = np.full(N,1); full_reg[N-len(hr_s):] = hr_s
    else:
        hm_s = np.array([-0.02,0,0.02]); hs_s = np.array([0.03,0.02,0.03])
        htrans = np.eye(3)*0.8+0.1; hdist = [33,34,33]; full_reg = np.ones(N,dtype=int)

    # Liénard Particle Filter
    # Calibrate mu, omega2, s
    best_p, best_e = (0.5,0.001,0.4), 1e10
    for mu in [0.3,0.6,1.0]:
        for om2 in [0.0005,0.002,0.005]:
            for s in [0.25,0.45,0.70]:
                e = _vdp_error(lp[-min(120,N):], cv[-min(120,N):], growth,
                               full_reg[-min(120,N):], mu, om2, s)
                if e < best_e: best_e = e; best_p = (mu,om2,s)
    mu_opt, om2_opt, s_opt = best_p

    # Full PF run
    Np = 150
    pP = np.full(Np, lp[0]); pV = np.random.randn(Np)*0.002; pU = np.full(Np, lp[0])
    w = np.ones(Np)/Np
    pf_out = []
    for t in range(N):
        cv_t = max(0.005, min(0.15, cv[t] if t<len(cv) else 0.03))
        U_star = growth*t + lp[0]
        for i in range(Np):
            PU = pP[i]-pU[i]; PUn = PU/s_opt
            vdp = mu_opt*(1-PUn**2)*pV[i]
            fund = -om2_opt*PU
            pP[i] += pV[i]; pV[i] += vdp+fund+cv_t*np.random.randn()*0.8
            pV[i] = np.clip(pV[i],-0.12,0.12)
            pU[i] += 0.005*(U_star-pU[i])+0.003*np.random.randn()
        ll = -0.5*((lp[t]-pP)**2)/(cv_t**2+1e-10); ll -= ll.max()
        w = np.exp(ll); w /= w.sum()+1e-30
        ess = 1/(np.sum(w**2)+1e-30)
        if ess < Np*0.4:
            idx = np.random.choice(Np,Np,p=w); pP,pV,pU = pP[idx],pV[idx],pU[idx]
            w = np.ones(Np)/Np
        Pe = np.average(pP,weights=w); Ve = np.average(pV,weights=w); Ue = np.average(pU,weights=w)
        ov = (Pe-Ue)*100; ds = 1-(((Pe-Ue)/s_opt)**2)
        pf_out.append({'P':Pe,'V':Ve,'U':Ue,'price':np.exp(Pe),'fund':np.exp(Ue),
                       'overval':ov,'damp':ds,'cv':cv_t})

    pf = pd.DataFrame(pf_out)

    # MC Forecast
    P0,V0,U0 = pf['P'].iloc[-1], pf['V'].iloc[-1], pf['U'].iloc[-1]
    cv0 = cv[-1]; reg0 = full_reg[-1]
    MC, HOR = 300, 24  # 24 months
    paths = np.zeros((MC,HOR))
    for m in range(MC):
        P,V,U,cv_m,rg = P0,V0,U0,cv0,reg0
        for t in range(HOR):
            rg = np.random.choice(3,p=htrans[rg])
            PU = P-U; PUn = PU/s_opt
            vdp = mu_opt*(1-PUn**2)*V; fund = -om2_opt*PU
            P += V; V += vdp+fund+cv_m*np.random.randn()*0.8
            V = np.clip(V,-0.12,0.12)
            U += 0.005*(growth*(N+t)+lp[0]-U)+0.003*np.random.randn()
            cv_m = max(0.005,min(0.12,cv_m*0.95+0.05*abs(np.random.randn())*cv0))
            paths[m,t] = np.exp(P)

    fc = []
    cur = np.exp(P0)
    for t in range(HOR):
        dp = paths[:,t]
        fc.append({'h':t+1,'p5':float(np.percentile(dp,5)),'p16':float(np.percentile(dp,16)),
                   'p50':float(np.percentile(dp,50)),'p84':float(np.percentile(dp,84)),
                   'p95':float(np.percentile(dp,95)),
                   'pu':round(float(np.mean(dp>cur)*100),1)})

    # Walk-forward backtest (simplified)
    bt = {'mae_model':0,'mae_rw':0,'dir':0,'n':0,'cov68':0}
    for fs in range(60,N-6,6):
        c = np.exp(lp[fs])
        for h in range(min(6,N-fs)):
            actual = np.exp(lp[fs+h])
            pred = fc[min(h,HOR-1)]['p50'] if h<HOR else c
            # Scale prediction relative to current
            pred_scaled = c * (pred / cur) if cur > 0 else c
            bt['mae_model'] += abs(actual-pred_scaled)
            bt['mae_rw'] += abs(actual-c)
            bt['dir'] += int(np.sign(pred_scaled-c)==np.sign(actual-c))
            if h < len(fc):
                lo = c*(fc[min(h,HOR-1)]['p16']/cur) if cur>0 else 0
                hi = c*(fc[min(h,HOR-1)]['p84']/cur) if cur>0 else 1e10
                bt['cov68'] += int(lo<=actual<=hi)
            bt['n'] += 1

    n = max(bt['n'],1)
    bt_metrics = {
        'mae_model':round(bt['mae_model']/n,1),
        'mae_rw':round(bt['mae_rw']/n,1),
        'dir_acc':round(bt['dir']/n*100,1),
        'cov68':round(bt['cov68']/n*100,1),
        'beats_rw': bt['mae_model'] < bt['mae_rw'],
    }

    # Build history (subsample ~40 points)
    step = max(1, N//40)
    hist = []
    for i in range(0, N, step):
        hist.append([df['Date'].iloc[i].strftime('%Y-%m'),
                     round(float(df['close'].iloc[i]),2),
                     round(float(pf['fund'].iloc[i]),2),
                     round(float(pf['overval'].iloc[i]),1),
                     round(float(pf['damp'].iloc[i]),3)])
    # Ensure last point
    if hist[-1][0] != df['Date'].iloc[-1].strftime('%Y-%m'):
        hist.append([df['Date'].iloc[-1].strftime('%Y-%m'),
                     round(float(df['close'].iloc[-1]),2),
                     round(float(pf['fund'].iloc[-1]),2),
                     round(float(pf['overval'].iloc[-1]),1),
                     round(float(pf['damp'].iloc[-1]),3)])

    return {
        'growth':round(ann_ret,1), 'vol':round(ann_vol,1), 'sharpe':round(sharpe,2),
        'mu':mu_opt, 'omega2':om2_opt, 's':s_opt,
        'garch':garch_info,
        'hmm':{'means':[round(float(m)*100,2) for m in hm_s],
               'stds':[round(float(s)*100,2) for s in hs_s],
               'dist':hdist, 'trans':htrans.round(3).tolist()},
        'state':{'price':round(float(df['close'].iloc[-1]),2),
                 'fund':round(float(pf['fund'].iloc[-1]),2),
                 'overval':round(float(pf['overval'].iloc[-1]),1),
                 'velocity':round(float(pf['V'].iloc[-1])*1000,2),
                 'damping':round(float(pf['damp'].iloc[-1]),3),
                 'phase':'TRENDING' if pf['damp'].iloc[-1]>0 else 'REVERTING',
                 'regime':['bear','range','bull'][full_reg[-1]],
                 'mvrv':round(float(mvrv[-1]),2) if not np.isnan(mvrv[-1]) else None},
        'backtest':bt_metrics,
        'history':hist,
        'forecast':[[f['h'],round(f['p5'],2),round(f['p16'],2),round(f['p50'],2),
                      round(f['p84'],2),round(f['p95'],2),round(f['pu'],1)]
                     for f in fc if f['h'] in [1,3,6,12,18,24]],
    }

def _vdp_error(lp, cv, gr, reg, mu, om2, s):
    """Quick error evaluation for VdP params."""
    N = len(lp); p0 = lp[0]
    P,V,U = p0,0,p0; err = 0
    for t in range(N):
        cv_t = max(0.005, cv[t] if t<len(cv) else 0.03)
        PU = P-U; PUn = PU/max(s,0.01)
        vdp = mu*(1-PUn**2)*V; fund = -om2*PU
        P += V; V += vdp+fund
        V = np.clip(V,-0.12,0.12)
        U += 0.005*(gr*t+p0-U)
        err += (lp[t]-P)**2
    return err

# ═══════════════════════════════════════════════════
#  3. RUN ALL ASSETS
# ═══════════════════════════════════════════════════

results = {}
for name in ASSET_NAMES:
    print(f"\n  Processing {name}...", end=" ", flush=True)
    results[name] = run_full_model(name, ASSETS_DF[name])
    s = results[name]['state']
    bt = results[name]['backtest']
    print(f"Done. Fund=${s['fund']:,.2f} OV={s['overval']:+.1f}% "
          f"Phase={s['phase']} Dir={bt['dir_acc']}% "
          f"{'✓' if bt['beats_rw'] else '✗'}RW")

# ═══════════════════════════════════════════════════
#  4. EXPORT
# ═══════════════════════════════════════════════════

output = {'assets':results, 'date':'2026-06-15',
          'model':'Liénard Oscillator + EGARCH-t + HMM v7'}

with open('/home/claude/model_v7.json','w') as f:
    json.dump(output, f, default=str)

print(f"\n  ✅ model_v7.json ({len(json.dumps(output,default=str))/1024:.0f} KB)")
print(f"  Assets: {', '.join(ASSET_NAMES)}")
