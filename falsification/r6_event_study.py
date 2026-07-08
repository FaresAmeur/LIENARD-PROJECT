"""
R6 — EVENT STUDY: CYCLE SCORE (MVRV zone x exogenous phase)
Pre-registered:
 T1. Before each major top (local max followed by >30% DD within 26w),
     what was the state (MVRV zone, phase d_t) in the 4 weeks prior?
 T2. 2x2 conditioning: forward 26w returns by (MVRV<2.4 vs >=2.4) x (d>0 vs d<=0).
     Prediction: cold+trending = best; hot+reverting = worst.
 T3. Cycle score S = z(MVRV inverted) + z(d): top vs bottom quintile forward 26w.
"""
import numpy as np, pandas as pd, json
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

mv = pd.read_csv('/home/claude/mvrv_real.csv', parse_dates=['time']).set_index('time')
mvw = mv.resample('W-FRI').last().dropna()
mvw = mvw[mvw.index>='2015-01-01']
lp = np.log(mvw['PriceUSD'].values); mvrv = mvw['MVRV'].values
U_exo = np.log(mvw['RealizedPrice'].values); N = len(mvw)

# Same exogenous PF as R3-C (identical seed/params → reproducible)
def run_pf(lp,U_exo,mu=0.6,om=0.01,s=0.25,Np=150,seed=42):
    rng=np.random.RandomState(seed)
    cv=pd.Series(np.diff(lp)).rolling(12,min_periods=4).std().values
    cv=np.where(np.isnan(cv),np.nanmean(cv[~np.isnan(cv)]),cv); cv=np.concatenate([[cv[0]],cv])
    pP=np.full(Np,lp[0]);pV=rng.randn(Np)*0.005;pU=np.full(Np,U_exo[0])+rng.randn(Np)*0.02
    w=np.ones(Np)/Np; d=[]
    for t in range(len(lp)):
        c=max(0.01,min(0.20,cv[t]))
        for i in range(Np):
            PU=pP[i]-pU[i]
            pP[i]+=pV[i];pV[i]+=mu*(1-(PU/s)**2)*pV[i]-om*PU+c*rng.randn()
            pV[i]=np.clip(pV[i],-0.15,0.15);pU[i]+=0.30*(U_exo[t]-pU[i])+0.003*rng.randn()
        ll=-0.5*((lp[t]-pP)**2)/(c**2+1e-10);ll-=ll.max();w=np.exp(ll);w/=w.sum()+1e-30
        if 1/(np.sum(w**2)+1e-30)<Np*0.35:
            idx=rng.choice(Np,Np,p=w);pP,pV,pU=pP[idx],pV[idx],pU[idx];w=np.ones(Np)/Np
        Pe=np.average(pP,weights=w);Ue=np.average(pU,weights=w)
        d.append(1-((Pe-Ue)/s)**2)
    return np.array(d)

print("Running exogenous-fundamental PF (596 weeks)...")
d = run_pf(lp, U_exo)

# T1: identify major tops (local max, >30% DD within next 26w)
print("\n─── T1: State in the 4 weeks BEFORE each major top ───")
tops=[]
for t in range(8, N-26):
    if lp[t]==max(lp[max(0,t-8):t+9]):
        dd=(np.exp(min(lp[t:t+26]))/np.exp(lp[t])-1)*100
        if dd<-30: tops.append((t,dd))
# dedupe (keep first of clusters)
tops=[tp for i,tp in enumerate(tops) if i==0 or tp[0]-tops[i-1][0]>13]
t1=[]
for t,dd in tops:
    pre_d=d[max(0,t-4):t]; pre_mv=mvrv[max(0,t-4):t]
    rev=(pre_d<=0).mean()*100
    t1.append({'date':str(mvw.index[t].date()),'price':round(float(np.exp(lp[t]))),
               'dd_next26w':round(dd,1),'mvrv_pre':round(float(pre_mv.mean()),2),
               'pct_reverting_pre':round(float(rev),0)})
    print(f"  {mvw.index[t].date()}  ${np.exp(lp[t]):>8,.0f}  DD26w={dd:+.0f}%  "
          f"MVRV_pre={pre_mv.mean():.2f}  reverting_pre={rev:.0f}%")
warn=np.mean([x['pct_reverting_pre'] for x in t1]) if t1 else 0
base_rev=(d<=0).mean()*100
print(f"  Avg reverting before tops: {warn:.0f}%  (unconditional: {base_rev:.0f}%)")

# T2: 2x2 conditioning
print("\n─── T2: Forward 26w return by (MVRV zone x phase) ───")
fwd=np.full(N,np.nan); fwd[:-26]=(lp[26:]-lp[:-26])*100
t2={}
for zn,zm in [("MVRV<2.4",mvrv<2.4),("MVRV>=2.4",mvrv>=2.4)]:
    for pn_,pm in [("trending",d>0),("reverting",d<=0)]:
        m=zm&pm&~np.isnan(fwd)
        if m.sum()>=5:
            t2[f"{zn} x {pn_}"]={'n':int(m.sum()),'ret':round(float(fwd[m].mean()),1),
                                 'pos':round(float((fwd[m]>0).mean()*100),0)}
            print(f"  {zn:10s} x {pn_:9s}: n={m.sum():3d}  ret26w={fwd[m].mean():+7.1f}%  P(↑)={(fwd[m]>0).mean()*100:.0f}%")

# T3: cycle score quintiles
print("\n─── T3: Cycle score S = z(-MVRV) + z(d), quintile spread ───")
S=-(mvrv-np.nanmean(mvrv))/np.nanstd(mvrv)+(d-np.nanmean(d))/np.nanstd(d)
v=~np.isnan(fwd)
q=np.nanquantile(S[v],[0.2,0.8])
lo,hi=fwd[v&(S<=q[0])],fwd[v&(S>=q[1])]
print(f"  Top quintile (best score):    n={len(hi)}  ret26w={hi.mean():+.1f}%  P(↑)={(hi>0).mean()*100:.0f}%")
print(f"  Bottom quintile (worst):      n={len(lo)}  ret26w={lo.mean():+.1f}%  P(↑)={(lo>0).mean()*100:.0f}%")
spread=hi.mean()-lo.mean()
# block permutation p-value
boots=[]
for b in range(2000):
    Sp=np.roll(S,np.random.randint(26,N-26))
    l2,h2=fwd[v&(Sp<=q[0])],fwd[v&(Sp>=q[1])]
    if len(l2)>5 and len(h2)>5: boots.append(h2.mean()-l2.mean())
pv=np.mean(np.abs(boots)>=abs(spread))
print(f"  Spread: {spread:+.1f}pp  |  circular-shift p-value: {pv:.3f}  "
      f"{'✓ significant' if pv<0.05 else '✗ n.s.'}")

# Current state
print(f"\n─── CURRENT STATE ({mvw.index[-1].date()}) ───")
print(f"  Price=${np.exp(lp[-1]):,.0f}  MVRV={mvrv[-1]:.2f}  phase={'TRENDING' if d[-1]>0 else 'REVERTING'} (d={d[-1]:+.2f})")
print(f"  Cycle score S={S[-1]:+.2f} (percentile {100*np.mean(S<=S[-1]):.0f})")

out={'t1_tops':t1,'t1_avg_warn':round(float(warn),0),'t1_base':round(float(base_rev),0),
     't2':t2,'t3':{'spread':round(float(spread),1),'p':round(float(pv),3),
     'top_q':{'ret':round(float(hi.mean()),1),'pos':round(float((hi>0).mean()*100),0)},
     'bot_q':{'ret':round(float(lo.mean()),1),'pos':round(float((lo>0).mean()*100),0)}},
     'current':{'mvrv':round(float(mvrv[-1]),2),'d':round(float(d[-1]),2),
                'score':round(float(S[-1]),2),'pctile':round(float(100*np.mean(S<=S[-1])),0)}}
json.dump(out,open('/home/claude/r6_results.json','w'),indent=2)
print("\n✅ r6_results.json")
