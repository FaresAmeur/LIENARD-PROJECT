"""
R1 — STEP 2: FROZEN v1.0 TEST ON REAL DATA
=============================================
Protocol:
  - Parameters FROZEN from paper Appendix B (calibrated on interpolated data)
  - Thresholds FROZEN from paper Table 5
  - NO grid search, NO re-tuning, NO adaptation
  - Entire real-data sample is out-of-sample w.r.t. all design choices
  - Same pipeline: EGARCH → HMM → PF → ensemble → phase filter → costs

FROZEN v1.0 PARAMETERS (from lienard_paper_v1, Appendix B & Table 5):
  SP500:   mu=0.3 omega2=0.010 s=0.25  thresholds=(0.35, 0.52)  cost=10bp
  Gold:    mu=0.3 omega2=0.010 s=0.25  thresholds=(0.35, 0.62)  cost=10bp
  Oil:     mu=0.3 omega2=0.010 s=0.25  thresholds=(0.35, 0.52)  cost=10bp
  EUR/USD: mu=0.3 omega2=0.005 s=0.25  thresholds=(0.48, 0.52)  cost=5bp
  SMA window: 52 weeks. Regime multipliers: [1.3, 0.7, 1.1]. Ensemble weights as v1.0.
"""
import numpy as np, pandas as pd, json
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

FROZEN = {
    'SP500':   {'mu':0.3,'om':0.010,'s':0.25,'th_dn':0.35,'th_up':0.52,'cost':0.0010},
    'Gold':    {'mu':0.3,'om':0.010,'s':0.25,'th_dn':0.35,'th_up':0.62,'cost':0.0010},
    'Oil':     {'mu':0.3,'om':0.010,'s':0.25,'th_dn':0.35,'th_up':0.52,'cost':0.0010},
    'EUR/USD': {'mu':0.3,'om':0.005,'s':0.25,'th_dn':0.48,'th_up':0.52,'cost':0.0005},
}

data = pd.read_csv('/home/claude/real_weekly.csv', parse_dates=['Date'])

print("═"*70)
print("  FROZEN v1.0 SPEC — REAL MARKET DATA (Oanda, 2015-2020)")
print("═"*70)

def fit_garch(rets):
    rp = rets*100
    for vt in ['EGARCH','Garch']:
        for dt in ['t','normal']:
            try:
                m = arch_model(pd.Series(rp),vol=vt,p=1,q=1,mean='Zero',dist=dt)
                f = m.fit(disp='off',show_warning=False)
                return np.concatenate([[f.conditional_volatility.iloc[0]],f.conditional_volatility.values])/100
            except: continue
    cv = pd.Series(rets).rolling(12,min_periods=4).std().values
    cv = np.where(np.isnan(cv), np.nanmean(cv[~np.isnan(cv)]) if (~np.isnan(cv)).any() else 0.02, cv)
    return np.concatenate([[cv[0]],cv])

def fit_hmm(rets):
    X = rets[~np.isnan(rets)].reshape(-1,1)
    best,sc = None,-np.inf
    for s in range(8):
        try:
            m = GaussianHMM(n_components=3,covariance_type='full',n_iter=200,random_state=s)
            m.fit(X)
            if m.score(X)>sc: best,sc = m,m.score(X)
        except: pass
    if not best: return np.ones(len(X),dtype=int), np.eye(3)*0.8+0.067
    hr = best.predict(X); means = best.means_.flatten(); order = np.argsort(means)
    rmap = {old:new for new,old in enumerate(order)}
    return np.array([rmap[r] for r in hr]), best.transmat_[order][:,order]

def run_pf(lp, cv, mvrv, reg, gr, mu, om, s, Np=100):
    N=len(lp); p0=lp[0]
    pP=np.full(Np,p0); pV=np.random.randn(Np)*0.005
    u0=np.log(np.exp(p0)/max(mvrv[0],0.3)) if not np.isnan(mvrv[0]) else p0
    pU=np.full(Np,u0)+np.random.randn(Np)*0.03; w=np.ones(Np)/Np; out=[]
    for t in range(N):
        cv_t=max(0.008,min(0.20,cv[t] if t<len(cv) else 0.05))
        mv=mvrv[t] if t<len(mvrv) and not np.isnan(mvrv[t]) else 1.5
        U_star=0.6*(lp[t]-np.log(max(mv,0.3)))+0.4*(gr*t+p0)
        me=mu*[1.3,0.7,1.1][reg[t] if t<len(reg) else 1]
        for i in range(Np):
            PU=pP[i]-pU[i]; PUn=PU/s
            pP[i]+=pV[i]; pV[i]+=me*(1-PUn**2)*pV[i]-om*PU+0.03*np.random.randn()+cv_t*np.random.randn()
            pV[i]=np.clip(pV[i],-0.15,0.15); pU[i]+=0.02*(U_star-pU[i])+0.005*np.random.randn()
        ll=-0.5*((lp[t]-pP)**2)/(cv_t**2+1e-10); ll-=ll.max()
        w=np.exp(ll); w/=w.sum()+1e-30
        if 1/(np.sum(w**2)+1e-30)<Np*0.35:
            idx=np.random.choice(Np,Np,p=w); pP,pV,pU=pP[idx],pV[idx],pU[idx]; w=np.ones(Np)/Np
        Pe=np.average(pP,weights=w); Ve=np.average(pV,weights=w); Ue=np.average(pU,weights=w)
        out.append({'P':Pe,'V':Ve,'U':Ue,'damp':1-((Pe-Ue)/s)**2})
    return out

results = {}
for name, frozen in FROZEN.items():
    print(f"\n{'━'*50}")
    print(f"  {name} — FROZEN: μ={frozen['mu']} ω²={frozen['om']} s={frozen['s']} "
          f"th=({frozen['th_dn']},{frozen['th_up']})")
    print(f"{'━'*50}")
    
    df = data[data['asset']==name].sort_values('Date').reset_index(drop=True)
    lp = np.log(df['close'].values).astype(float)
    N = len(lp)
    rets = np.diff(lp)
    gr = (lp[-1]-lp[0])/N
    
    sma = pd.Series(np.exp(lp)).rolling(52, min_periods=26).mean().values
    mvrv = np.exp(lp)/(sma+1e-8)
    cv = fit_garch(rets)
    hr, htrans = fit_hmm(rets)
    freg = np.full(N,1); freg[N-len(hr):] = hr
    
    mu, om, s = frozen['mu'], frozen['om'], frozen['s']
    th_up, th_dn = frozen['th_up'], frozen['th_dn']
    cost = frozen['cost']
    
    # Walk-forward — same protocol as v1.0
    TR, TE, ST, NM = 52, 8, 4, 80
    rows = []
    trail_rw, trail_ens = [], []
    
    for fs in range(TR, N-TE, ST):
        tlp = lp[fs-TR:fs]; tcv = cv[max(0,fs-TR):fs]
        tmv = mvrv[fs-TR:fs]; treg = freg[fs-TR:fs]
        trets = np.diff(tlp); cur = np.exp(lp[fs-1])
        
        pft = run_pf(tlp, tcv, tmv, treg, gr, mu, om, s, 40)
        P0,V0,U0 = pft[-1]['P'], pft[-1]['V'], pft[-1]['U']
        d0 = pft[-1]['damp']
        cv0 = tcv[-1] if len(tcv)>0 else 0.03
        r0 = treg[-1] if len(treg)>0 else 1
        drift = np.mean(trets) if len(trets)>0 else 0
        ar1 = np.corrcoef(trets[:-1],trets[1:])[0,1] if len(trets)>2 else 0
        
        if len(trail_rw)>=4 and np.mean(trail_ens[-4:])<np.mean(trail_rw[-4:]):
            wr,wa,wv = 0.20,0.30,0.50
        else:
            wr,wa,wv = 0.45,0.30,0.25
        
        for h in range(min(TE, N-fs)):
            actual = np.exp(lp[fs+h]); prw = cur
            par = cur*np.exp(drift*(h+1)+ar1*trets[-1]*(0.8**h) if len(trets)>0 else 0)
            mc = np.zeros(NM)
            for m in range(NM):
                P,V,U = P0,V0,U0; cvm = cv0; rg = r0
                for step in range(h+1):
                    rg = np.random.choice(3, p=htrans[rg])
                    mm = mu*[1.3,0.7,1.1][rg]
                    PU = P-U; PUn = PU/s
                    P += V; V += mm*(1-PUn**2)*V-om*PU+0.03*np.random.randn()+cvm*np.random.randn()
                    V = np.clip(V,-0.15,0.15)
                    U += 0.02*(gr*(TR+step)+tlp[0]-U)+0.005*np.random.randn()
                    cvm = max(0.008, cvm*0.97+0.03*abs(np.random.randn())*cv0)
                mc[m] = np.exp(P)
            pvdp = np.median(mc)
            pens = wr*prw+wa*par+wv*pvdp
            pu = wr*0.5+wa*(1 if par>cur else 0)+wv*np.mean(mc>cur)
            rows.append({'fold':fs,'h':h+1,'actual':actual,'cur':cur,
                'pens':pens,'prw':prw,'pu':pu,'d':d0,
                'ph':'T' if d0>0 else 'R',
                'err_e':abs(actual-pens),'err_rw':abs(actual-prw)})
        h1s = [r for r in rows[-TE:] if r['h']==1]
        if h1s: trail_rw.append(h1s[0]['err_rw']); trail_ens.append(h1s[0]['err_e'])
    
    bt = pd.DataFrame(rows)
    bt['dir'] = (np.sign(bt['pens']-bt['cur'])==np.sign(bt['actual']-bt['cur'])).astype(int)
    bt['ret'] = np.log(bt['actual']/bt['cur'])
    
    # ── Metrics ──
    h1 = bt[bt['h']==1].sort_values('fold').copy()
    n_folds = bt['fold'].nunique()
    
    # Phase-conditioned accuracy
    tr = h1[h1['ph']=='T']; rv = h1[h1['ph']=='R']
    print(f"  Folds: {n_folds}  |  W+1 obs: {len(h1)} ({len(tr)} trending, {len(rv)} reverting)")
    if len(tr)>3:
        ratio_t = tr['err_e'].mean()/(tr['err_rw'].mean()+1e-10)
        print(f"  TRENDING:  dir={tr['dir'].mean()*100:.1f}%  MAE ratio={ratio_t:.3f} "
              f"{'✓ beats RW' if ratio_t<1 else '✗'}")
    if len(rv)>3:
        ratio_r = rv['err_e'].mean()/(rv['err_rw'].mean()+1e-10)
        print(f"  REVERTING: dir={rv['dir'].mean()*100:.1f}%  MAE ratio={ratio_r:.3f}")
    
    # ── FROZEN strategy ──
    h1['sig'] = 0
    mask_t = h1['ph']=='T'
    h1.loc[(h1['pu']>th_up)&mask_t,'sig'] = 1
    h1.loc[(h1['pu']<th_dn)&mask_t,'sig'] = -1
    h1['sret'] = h1['sig']*h1['ret'] - abs(h1['sig'])*cost
    v = h1.dropna(subset=['sret','ret'])
    
    strat = {}
    if len(v)>10 and v['sret'].std()>0:
        sr = v['sret'].mean()/v['sret'].std()*np.sqrt(52)
        sr_b = v['ret'].mean()/(v['ret'].std()+1e-10)*np.sqrt(52)
        cm = (1+v['sret']).cumprod(); cb = (1+v['ret']).cumprod()
        dd = (cm/cm.cummax()-1).min()*100
        act = v['sig']!=0
        wr_ = (v['sret'][act]>0).mean()*100 if act.sum()>0 else 0
        n_trades = int(act.sum())
        strat = {'sharpe':round(float(sr),2),'sharpe_bnh':round(float(sr_b),2),
                 'ret':round(float((cm.iloc[-1]-1)*100),1),
                 'ret_bnh':round(float((cb.iloc[-1]-1)*100),1),
                 'dd':round(float(dd),1),'win':round(float(wr_),1),
                 'active_pct':round(float(act.mean()*100),0),'n_trades':n_trades}
        print(f"\n  ┌─ FROZEN STRATEGY (real data, net of {cost*1e4:.0f}bp costs) ─┐")
        print(f"  │ Sharpe:   {sr:>7.2f}   (B&H: {sr_b:.2f})")
        print(f"  │ Return:   {strat['ret']:>6.1f}%   (B&H: {strat['ret_bnh']}%)")
        print(f"  │ Max DD:   {strat['dd']:>6.1f}%")
        print(f"  │ Win rate: {wr_:>6.1f}%   ({n_trades} trades, {strat['active_pct']:.0f}% active)")
        print(f"  └{'─'*45}┘")
    
    results[name] = {
        'n_weeks': int(N), 'n_folds': int(n_folds),
        'real_vol': round(float(np.std(rets)*np.sqrt(52)*100),1),
        'frozen_params': frozen,
        'phase_split': {'trending_n': int(len(tr)), 'reverting_n': int(len(rv)),
                        'trending_dir': round(float(tr['dir'].mean()*100),1) if len(tr)>3 else None,
                        'trending_ratio': round(float(tr['err_e'].mean()/(tr['err_rw'].mean()+1e-10)),3) if len(tr)>3 else None},
        'strategy': strat,
    }

# ═══ VERDICT ═══
print(f"\n{'═'*70}")
print(f"  R1 VERDICT — FROZEN v1.0 ON REAL DATA vs INTERPOLATED CLAIMS")
print(f"{'═'*70}")
print(f"\n  {'Asset':10s} │ {'Sharpe REAL':>11s} │ {'Sharpe v1.0':>11s} │ {'Survives?':>9s}")
print(f"  {'─'*10} │ {'─'*11} │ {'─'*11} │ {'─'*9}")
V10_CLAIMS = {'SP500':2.76,'Gold':3.62,'Oil':2.68,'EUR/USD':3.78}
for name in FROZEN:
    real_sr = results[name]['strategy'].get('sharpe','—')
    claim = V10_CLAIMS[name]
    if isinstance(real_sr,(int,float)):
        verdict = '✓ YES' if real_sr>1.0 else ('~ PARTIAL' if real_sr>0 else '✗ NO')
    else:
        verdict = '?'
    print(f"  {name:10s} │ {real_sr:>11} │ {claim:>11.2f} │ {verdict:>9s}")

with open('/home/claude/r1_frozen_results.json','w') as f:
    json.dump(results, f, default=str, indent=2)
print(f"\n  ✅ r1_frozen_results.json saved")
