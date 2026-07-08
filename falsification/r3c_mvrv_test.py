"""
R3-C — TRUE ON-CHAIN MVRV AS EXOGENOUS FUNDAMENTAL
====================================================
Data: Coin Metrics community (real on-chain, 2010-2026, no interpolation)
      MVRV = MarketCap / RealizedCap ; RealizedPrice = Price / MVRV
Pre-registered tests (fixed BEFORE looking at results):
  T1. Cycle anchor: forward returns (13/26/52w) conditional on MVRV zones
      Zones fixed from literature: <1.0 accumulation, 1-2.4 neutral, >2.4 hot, >3.5 top
  T2. Circuit with EXOGENOUS U(t)=log(RealizedPrice): does the phase variable
      d_t discriminate again on REAL weekly BTC data? (R1 showed price-only
      phase degenerated). Compare W+1 direction: trending vs reverting weeks.
  T3. Full circuit 2020-2024: U=realized price + F=funding rate vs price-only.
      W+1 direction accuracy, same seeds, same protocol.
"""
import numpy as np, pandas as pd, json
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

print("═"*70)
print("  R3-C — TRUE ON-CHAIN MVRV (Coin Metrics, 2010-2026)")
print("═"*70)

mv = pd.read_csv('/home/claude/mvrv_real.csv', parse_dates=['time'])
mv = mv.set_index('time')
mvw = mv.resample('W-FRI').last().dropna()
mvw = mvw[mvw.index >= '2015-01-01']
lp = np.log(mvw['PriceUSD'].values)
mvrv = mvw['MVRV'].values
rp = mvw['RealizedPrice'].values
N = len(mvw)
print(f"  Weekly: {N} weeks ({mvw.index[0].date()} → {mvw.index[-1].date()})")
print(f"  Current: Price=${mvw['PriceUSD'].iloc[-1]:,.0f} MVRV={mvrv[-1]:.2f} RealizedPrice=${rp[-1]:,.0f}")

# ═══ T1: MVRV zones → forward returns ═══
print(f"\n{'─'*60}\n  T1 — Forward returns by MVRV zone (literature thresholds)\n{'─'*60}")
zones = [("MVRV < 1.0 (accumulation)", mvrv < 1.0),
         ("1.0-2.4 (neutre)", (mvrv>=1.0)&(mvrv<2.4)),
         ("2.4-3.5 (chaud)", (mvrv>=2.4)&(mvrv<3.5)),
         ("MVRV > 3.5 (top)", mvrv >= 3.5)]
t1 = {}
for h in [13, 26, 52]:
    fwd = np.full(N, np.nan)
    fwd[:-h] = (lp[h:] - lp[:-h]) * 100
    print(f"\n  Horizon {h} semaines:")
    t1[h] = {}
    for name, mask in zones:
        m = mask & ~np.isnan(fwd)
        if m.sum() >= 5:
            mean_r = fwd[m].mean(); pos = (fwd[m]>0).mean()*100
            print(f"    {name:28s}: n={m.sum():3d}  ret={mean_r:+7.1f}%  P(↑)={pos:.0f}%")
            t1[h][name] = {'n':int(m.sum()),'ret':round(float(mean_r),1),'pos':round(float(pos),0)}

# ═══ T2: Circuit with exogenous U = log(RealizedPrice) ═══
print(f"\n{'─'*60}\n  T2 — Phase variable with EXOGENOUS fundamental (all real data)\n{'─'*60}")

def run_pf_exo(lp, U_exo, forcing=None, mu=0.6, om=0.01, s=0.25, k_force=0.005, Np=150, seed=42):
    rng = np.random.RandomState(seed)
    N = len(lp)
    cvs = pd.Series(np.diff(lp)).rolling(12, min_periods=4).std().values
    cvs = np.where(np.isnan(cvs), np.nanmean(cvs[~np.isnan(cvs)]), cvs)
    cvs = np.concatenate([[cvs[0]], cvs])
    pP=np.full(Np,lp[0]); pV=rng.randn(Np)*0.005
    pU=np.full(Np,U_exo[0])+rng.randn(Np)*0.02; w=np.ones(Np)/Np
    states=[]
    for t in range(N):
        cv_t=max(0.01,min(0.20,cvs[t]))
        F_t = -k_force*forcing[t] if forcing is not None and t<len(forcing) and not np.isnan(forcing[t]) else 0.0
        for i in range(Np):
            PU=pP[i]-pU[i]; PUn=PU/s
            pP[i]+=pV[i]
            pV[i]+=mu*(1-PUn**2)*pV[i]-om*PU+F_t+cv_t*rng.randn()
            pV[i]=np.clip(pV[i],-0.15,0.15)
            # U anchored HARD to exogenous realized price (alpha=0.30, tight)
            pU[i]+=0.30*(U_exo[t]-pU[i])+0.003*rng.randn()
        ll=-0.5*((lp[t]-pP)**2)/(cv_t**2+1e-10); ll-=ll.max()
        w=np.exp(ll); w/=w.sum()+1e-30
        if 1/(np.sum(w**2)+1e-30)<Np*0.35:
            idx=rng.choice(Np,Np,p=w); pP,pV,pU=pP[idx],pV[idx],pU[idx]; w=np.ones(Np)/Np
        Pe=np.average(pP,weights=w); Ve=np.average(pV,weights=w); Ue=np.average(pU,weights=w)
        states.append({'V':Ve,'d':1-((Pe-Ue)/s)**2,'ov':(Pe-Ue)*100})
    return states

U_exo = np.log(rp)
st = run_pf_exo(lp, U_exo)
rets_next = np.append(np.diff(lp), np.nan)

d_arr = np.array([s['d'] for s in st])
v_arr = np.array([s['V'] for s in st])
n_trend = (d_arr>0).sum(); n_rev = (d_arr<=0).sum()
print(f"  Phase distribution: {n_trend} trending ({n_trend/N*100:.0f}%), {n_rev} reverting ({n_rev/N*100:.0f}%)")
print(f"  → {'✓ discriminates (vs 96% degenerate in R1)' if 0.15 < n_trend/N < 0.85 else '✗ still degenerate'}")

t2 = {}
for label, mask in [("TRENDING (d>0)", d_arr>0), ("REVERTING (d<=0)", d_arr<=0)]:
    m = mask[:-1] & ~np.isnan(rets_next[:-1]) & (v_arr[:-1]!=0)
    if m.sum()>10:
        correct = (np.sign(v_arr[:-1][m])==np.sign(rets_next[:-1][m])).mean()*100
        print(f"  {label:18s}: n={m.sum():3d}  W+1 direction = {correct:.1f}%")
        t2[label] = {'n':int(m.sum()),'dir':round(float(correct),1)}

# ═══ T3: Full circuit 2020-2024 (U=realized + F=funding) ═══
print(f"\n{'─'*60}\n  T3 — Full circuit vs price-only (2020-2024 overlap)\n{'─'*60}")
fr = pd.read_csv('/home/claude/funding_data/historical-funding-rates-fetcher-main/data/BTC-USDT/BTC-USDT_binance_2020-01-01_2024-01-01_funding_history.csv')
fr['Date'] = pd.to_datetime(fr['Date'])
frw = fr.set_index('Date')['Funding Rate'].resample('W-FRI').mean()

sub = mvw[(mvw.index>='2020-01-01')&(mvw.index<='2024-01-05')]
common = sub.index.intersection(frw.index)
lp3 = np.log(sub.loc[common,'PriceUSD'].values)
U3 = np.log(sub.loc[common,'RealizedPrice'].values)
f3 = frw.loc[common].values
f3z = (f3-np.nanmean(f3))/(np.nanstd(f3)+1e-12)
r3n = np.append(np.diff(lp3), np.nan)
print(f"  Window: {len(common)} weeks")

def dir_acc(states, rn):
    v = np.array([s['V'] for s in states])
    m = ~np.isnan(rn[:-1]) & (v[:-1]!=0)
    return (np.sign(v[:-1][m])==np.sign(rn[:-1][m])).mean()*100

# Price-only baseline: endogenous SMA fundamental
sma = pd.Series(np.exp(lp3)).rolling(52,min_periods=26).mean().values
U_endo = np.log(np.where(np.isnan(sma), np.exp(lp3), sma))
configs = {
    'price-only (U=SMA)': run_pf_exo(lp3, U_endo, None, k_force=0),
    'U=realized only': run_pf_exo(lp3, U3, None, k_force=0),
    'F=funding only (U=SMA)': run_pf_exo(lp3, U_endo, f3z, k_force=0.005),
    'FULL: U=realized + F=funding': run_pf_exo(lp3, U3, f3z, k_force=0.005),
}
t3 = {}
for name, states in configs.items():
    da = dir_acc(states, r3n)
    t3[name] = round(float(da),1)
    print(f"  {name:32s}: W+1 dir = {da:.1f}%")

# ═══ VERDICT ═══
print(f"\n{'═'*70}\n  R3-C VERDICT\n{'═'*70}")
acc13 = t1[26].get("MVRV < 1.0 (accumulation)",{})
t1_pass = acc13.get('pos',0) >= 70
t2_pass = 0.15 < n_trend/N < 0.85 and t2.get("TRENDING (d>0)",{}).get('dir',50) > 53
t3_pass = t3['FULL: U=realized + F=funding'] > t3['price-only (U=SMA)'] + 2
print(f"  T1 (MVRV cycle anchor):   {'✓ PASS' if t1_pass else '✗ FAIL'}")
print(f"  T2 (phase discriminates): {'✓ PASS' if t2_pass else '✗ FAIL'}")
print(f"  T3 (full circuit > price-only): {'✓ PASS' if t3_pass else '✗ FAIL'} "
      f"({t3['price-only (U=SMA)']}% → {t3['FULL: U=realized + F=funding']}%)")

out = {'coverage':f"{mvw.index[0].date()} to {mvw.index[-1].date()}",
       'current':{'price':round(float(mvw['PriceUSD'].iloc[-1])),'mvrv':round(float(mvrv[-1]),3),
                  'realized_price':round(float(rp[-1]))},
       't1':t1,'t2':t2,'t3':t3,
       'phase_dist':{'trending_pct':round(n_trend/N*100,1)},
       'verdict':{'t1':bool(t1_pass),'t2':bool(t2_pass),'t3':bool(t3_pass)}}
with open('/home/claude/r3c_results.json','w') as f:
    json.dump(out,f,indent=2,default=str)
print(f"\n  ✅ r3c_results.json")
