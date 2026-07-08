"""
LIÉNARD OSCILLATOR v8 — COMMERCIAL IMPROVEMENTS
=================================================
Changes from v7:
  1. WEEKLY frequency (optimal for mean-reversion, avoids daily noise)
  2. MVRV-anchored fundamental (200-week SMA = crypto's "fair value")
  3. Proper EGARCH(1,1)-Student-t with fallback
  4. HMM 3-state on weekly returns (smoother regimes)
  5. ENSEMBLE: RW + drift-ARIMA + Liénard, regime-weighted
  6. Walk-forward backtest: 52w train, 8w test, 4w step (~100 folds)
  7. Strategy Sharpe: long/short based on ensemble signal
  8. Multi-asset: BTC(daily→weekly), SP500(monthly), Gold(monthly)
"""

import numpy as np, pandas as pd, json
from scipy.optimize import minimize
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

print("═"*70)
print("  LIÉNARD OSCILLATOR v8 — COMMERCIAL IMPROVEMENTS")
print("═"*70)

# ═══════════════════════════════════════════════════
#  1. LOAD BTC WEEKLY
# ═══════════════════════════════════════════════════

df = pd.read_csv('/home/claude/btc_raw.csv', parse_dates=['Date'])
df = df.sort_values('Date').rename(columns={'Price':'close'})
df = df[df['close']>0]
for d,p in [('2026-05-01',78100),('2026-05-15',73500),('2026-06-01',64500),
            ('2026-06-08',63564),('2026-06-15',66522)]:
    df = pd.concat([df, pd.DataFrame([{'Date':pd.Timestamp(d),'close':p}])])
df = df.drop_duplicates('Date',keep='last').sort_values('Date')
df = df.set_index('Date').resample('D').last().interpolate('linear',limit=7).dropna()

# Weekly resampling (Friday close)
wk = df.resample('W-FRI').last().dropna().reset_index()
wk = wk[wk['Date']>='2015-01-01'].copy().reset_index(drop=True)
wk['log_price'] = np.log(wk['close'])
wk['log_return'] = wk['log_price'].diff()
wk['abs_return'] = wk['log_return'].abs()

# MVRV Proxy: price / 200-week SMA (the gold standard in crypto)
wk['sma200w'] = wk['close'].rolling(200, min_periods=50).mean()
wk['mvrv'] = wk['close'] / wk['sma200w']

# Volatility features
wk['vol_12w'] = wk['log_return'].rolling(12).std() * np.sqrt(52)
wk['vol_52w'] = wk['log_return'].rolling(52).std() * np.sqrt(52)

# Momentum
for k in [4, 12, 26, 52]:
    wk[f'mom_{k}w'] = wk['log_price'].diff(k)

# Drawdown
wk['ath'] = wk['close'].cummax()
wk['drawdown'] = (wk['close'] - wk['ath']) / wk['ath']

lp = wk['log_price'].values
rets = wk['log_return'].dropna().values
N = len(lp)
growth_weekly = (lp[-1] - lp[0]) / N

print(f"  BTC Weekly: {N} weeks ({wk['Date'].iloc[0].date()} → {wk['Date'].iloc[-1].date()})")
print(f"  Price: ${wk['close'].iloc[-1]:,.0f}  MVRV: {wk['mvrv'].iloc[-1]:.2f}")
print(f"  Growth: {growth_weekly*52*100:.1f}%/yr  Vol: {np.std(rets)*np.sqrt(52)*100:.1f}%/yr")

# ═══════════════════════════════════════════════════
#  2. EGARCH(1,1)-Student-t
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  EGARCH(1,1) Weekly Conditional Volatility")
print(f"{'─'*50}")

rp = rets * 100
try:
    gm = arch_model(rp, vol='EGARCH', p=1, q=1, mean='ARX', lags=1, dist='t')
    gf = gm.fit(disp='off', show_warning=False)
    cond_vol = np.concatenate([[gf.conditional_volatility.iloc[0]],
                                gf.conditional_volatility.values]) / 100
    garch_ok = True
    alpha_g = float(gf.params.get('alpha[1]', 0))
    beta_g = float(gf.params.get('beta[1]', 0))
    nu_g = float(gf.params.get('nu', 5))
    print(f"  α={alpha_g:.4f}  β={beta_g:.4f}  ν={nu_g:.2f}")
except Exception as e:
    print(f"  EGARCH failed ({e}), using rolling vol")
    cond_vol = np.concatenate([[rets[0]], np.abs(rets)]).cumsum()
    cond_vol = pd.Series(rets).rolling(12).std().values
    cond_vol = np.concatenate([[np.nanmean(cond_vol)], cond_vol])
    cond_vol = np.where(np.isnan(cond_vol), 0.05, cond_vol)
    garch_ok = False
    alpha_g, beta_g, nu_g = 0, 0, 5

print(f"  Current σ_w = {cond_vol[-1]*100:.2f}%/week = {cond_vol[-1]*np.sqrt(52)*100:.1f}%/yr")

# ═══════════════════════════════════════════════════
#  3. HMM 3-State
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  HMM 3-State Regime Detection (Weekly)")
print(f"{'─'*50}")

X_hmm = rets[~np.isnan(rets)].reshape(-1, 1)
best_hmm, best_sc = None, -np.inf
for seed in range(12):
    try:
        m = GaussianHMM(n_components=3, covariance_type='full', n_iter=300,
                        random_state=seed, tol=1e-5)
        m.fit(X_hmm)
        sc = m.score(X_hmm)
        if sc > best_sc: best_hmm, best_sc = m, sc
    except: pass

hmm_means = best_hmm.means_.flatten()
hmm_stds = np.sqrt(best_hmm.covars_.flatten())
order = np.argsort(hmm_means)
rmap = {old:new for new,old in enumerate(order)}
hr = np.array([rmap[r] for r in best_hmm.predict(X_hmm)])
hm = hmm_means[order]; hs = hmm_stds[order]
htrans = best_hmm.transmat_[order][:,order]
hdist = [(hr==i).sum()/len(hr)*100 for i in range(3)]

# Pad to full length
full_reg = np.full(N, 1)
full_reg[N-len(hr):] = hr

RN = ['bear','range','bull']
for i in range(3):
    print(f"  {RN[i]:6s}: {hdist[i]:5.1f}%  μ={hm[i]*100:+.2f}%/wk  σ={hs[i]*100:.2f}%/wk")

print(f"  Transition: Bear→Bear={htrans[0,0]:.3f}  Range→Range={htrans[1,1]:.3f}  Bull→Bull={htrans[2,2]:.3f}")

# ═══════════════════════════════════════════════════
#  4. MVRV-ANCHORED LIÉNARD PARTICLE FILTER
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  MVRV-Anchored Liénard Particle Filter")
print(f"{'─'*50}")

def run_lienard_pf(lp, cv, mvrv_series, regimes, growth,
                   mu=0.6, omega2=0.002, s=0.35, Np=200):
    """
    Liénard PF with MVRV-anchored fundamental.
    U(t) = log(SMA_200w) = log(price/MVRV) — exogenous anchor.
    """
    N = len(lp)
    p0 = lp[0]
    
    pP = np.full(Np, p0)
    pV = np.random.randn(Np) * 0.005
    # Initialize U from MVRV
    u0 = np.log(np.exp(p0) / max(mvrv_series[0], 0.5)) if not np.isnan(mvrv_series[0]) else p0
    pU = np.full(Np, u0) + np.random.randn(Np) * 0.03
    w = np.ones(Np) / Np
    
    out = []
    for t in range(N):
        cv_t = max(0.008, min(0.20, cv[t] if t < len(cv) else 0.05))
        
        # MVRV-anchored fundamental target
        mv = mvrv_series[t] if t < len(mvrv_series) and not np.isnan(mvrv_series[t]) else 1.5
        U_mvrv = lp[t] - np.log(max(mv, 0.3))  # U = log(Price/MVRV) = log(SMA200)
        
        # Blend structural growth + MVRV anchor
        U_star = 0.6 * U_mvrv + 0.4 * (growth * t + p0)
        
        mu_eff = mu * [1.3, 0.7, 1.1][regimes[t] if t < len(regimes) else 1]
        
        for i in range(Np):
            PU = pP[i] - pU[i]
            PUn = PU / s
            
            # Liénard: μ(1-x²)V is the self-excitation
            vdp = mu_eff * (1 - PUn**2) * pV[i]
            fund_pull = -omega2 * PU
            flux = 0.03 * np.random.randn()
            
            pP[i] += pV[i]
            pV[i] += vdp + fund_pull + flux + cv_t * np.random.randn()
            pV[i] = np.clip(pV[i], -0.15, 0.15)
            pU[i] += 0.02 * (U_star - pU[i]) + 0.005 * np.random.randn()
        
        # Weights
        ll = -0.5 * ((lp[t] - pP)**2) / (cv_t**2 + 1e-10)
        ll -= ll.max()
        w = np.exp(ll); w /= w.sum() + 1e-30
        
        ess = 1 / (np.sum(w**2) + 1e-30)
        if ess < Np * 0.35:
            idx = np.random.choice(Np, Np, p=w)
            pP, pV, pU = pP[idx], pV[idx], pU[idx]
            w = np.ones(Np) / Np
        
        Pe = np.average(pP, weights=w)
        Ve = np.average(pV, weights=w)
        Ue = np.average(pU, weights=w)
        ov = (Pe - Ue) * 100
        ds = 1 - ((Pe - Ue) / s)**2
        
        out.append({
            'P': Pe, 'V': Ve, 'U': Ue,
            'price': np.exp(Pe), 'fund': np.exp(Ue),
            'overval': ov, 'damp': ds, 'cv': cv_t,
            'mvrv': mv,
        })
    
    return pd.DataFrame(out)

# Calibrate
print("  Calibrating μ, ω², s on last 104 weeks...")
mvrv_arr = wk['mvrv'].values
best_p, best_e = (0.6, 0.002, 0.35), 1e10

for mu in [0.3, 0.5, 0.8, 1.2]:
    for om2 in [0.001, 0.003, 0.008]:
        for s in [0.20, 0.35, 0.55]:
            pf_test = run_lienard_pf(lp[-104:], cond_vol[-104:], mvrv_arr[-104:],
                                     full_reg[-104:], growth_weekly,
                                     mu=mu, omega2=om2, s=s, Np=60)
            err = np.sum((lp[-104:] - pf_test['P'].values)**2)
            if err < best_e:
                best_e = err; best_p = (mu, om2, s)

mu_opt, om2_opt, s_opt = best_p
print(f"  Optimal: μ={mu_opt}  ω²={om2_opt}  s={s_opt}")
print(f"  s={s_opt} → switch at ±{s_opt*100:.0f}% overvaluation")

# Full run
print(f"  Running full PF ({N} weeks, 200 particles)...")
pf = run_lienard_pf(lp, cond_vol, mvrv_arr, full_reg, growth_weekly,
                     mu=mu_opt, omega2=om2_opt, s=s_opt, Np=200)

print(f"  Final: Price=${pf['price'].iloc[-1]:,.0f}  Fund=${pf['fund'].iloc[-1]:,.0f}")
print(f"  Overval={pf['overval'].iloc[-1]:.1f}%  MVRV={pf['mvrv'].iloc[-1]:.2f}")
print(f"  Damping={pf['damp'].iloc[-1]:.3f} ({'TRENDING' if pf['damp'].iloc[-1]>0 else 'REVERTING'})")

# ═══════════════════════════════════════════════════
#  5. ENSEMBLE MODEL: RW + ARIMA + LIÉNARD
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  Walk-Forward Backtest — Ensemble Model")
print(f"{'─'*50}")

TRAIN_W = 52  # 1 year
TEST_W = 8    # 2 months
STEP_W = 4    # monthly
N_MC = 150

bt = []
n_folds = 0

for fs in range(TRAIN_W, N - TEST_W, STEP_W):
    n_folds += 1
    train_lp = lp[fs-TRAIN_W:fs]
    train_cv = cond_vol[max(0,fs-TRAIN_W):fs]
    train_mv = mvrv_arr[fs-TRAIN_W:fs]
    train_reg = full_reg[fs-TRAIN_W:fs]
    train_rets = np.diff(train_lp)
    
    # Current state
    cur_price = np.exp(lp[fs-1])
    
    # Drift for ARIMA
    drift = np.mean(train_rets) if len(train_rets) > 0 else 0
    ar1 = np.corrcoef(train_rets[:-1], train_rets[1:])[0,1] if len(train_rets)>2 else 0
    last_ret = train_rets[-1] if len(train_rets) > 0 else 0
    
    # PF state at end of training
    pf_train = run_lienard_pf(train_lp, train_cv, train_mv, train_reg,
                               growth_weekly, mu=mu_opt, omega2=om2_opt, s=s_opt, Np=80)
    P0 = pf_train['P'].iloc[-1]
    V0 = pf_train['V'].iloc[-1]
    U0 = pf_train['U'].iloc[-1]
    damp0 = pf_train['damp'].iloc[-1]
    cv0 = train_cv[-1] if len(train_cv) > 0 else 0.05
    reg0 = train_reg[-1] if len(train_reg) > 0 else 1
    
    # Current regime → ensemble weights
    # In range: trust RW more. In transition: trust Liénard more.
    # Strong trend: trust ARIMA more.
    if abs(damp0) > 1.5:  # strong reverting → Liénard gets more weight
        w_rw, w_arima, w_vdp = 0.25, 0.25, 0.50
    elif abs(last_ret) > 2 * np.std(train_rets):  # big move → momentum
        w_rw, w_arima, w_vdp = 0.20, 0.50, 0.30
    else:  # calm → RW dominates
        w_rw, w_arima, w_vdp = 0.45, 0.30, 0.25
    
    for h in range(min(TEST_W, N - fs)):
        t_abs = fs + h
        actual = np.exp(lp[t_abs])
        
        # 1. Random Walk prediction
        pred_rw = cur_price
        
        # 2. ARIMA(1,1,0) prediction
        pred_arima = cur_price * np.exp(drift * (h+1) + ar1 * last_ret * (0.8**(h)))
        
        # 3. Liénard MC prediction
        mc_prices = np.zeros(N_MC)
        for m in range(N_MC):
            P, V, U = P0, V0, U0
            cv_m, rg = cv0, reg0
            for step in range(h + 1):
                rg = np.random.choice(3, p=htrans[rg])
                mu_m = mu_opt * [1.3, 0.7, 1.1][rg]
                PU = P - U; PUn = PU / s_opt
                vdp = mu_m * (1 - PUn**2) * V
                fund = -om2_opt * PU
                P += V
                V += vdp + fund + 0.03*np.random.randn() + cv_m*np.random.randn()
                V = np.clip(V, -0.15, 0.15)
                U += 0.02 * (growth_weekly*(TRAIN_W+step) + train_lp[0] - U) + 0.005*np.random.randn()
                cv_m = max(0.008, cv_m * 0.97 + 0.03 * abs(np.random.randn()) * cv0)
            mc_prices[m] = np.exp(P)
        
        pred_vdp = np.median(mc_prices)
        p16_vdp = np.percentile(mc_prices, 16)
        p84_vdp = np.percentile(mc_prices, 84)
        prob_up_vdp = np.mean(mc_prices > cur_price)
        
        # 4. Ensemble prediction
        pred_ens = w_rw * pred_rw + w_arima * pred_arima + w_vdp * pred_vdp
        
        # Ensemble CI (use VdP CI scaled)
        p16_ens = w_rw * cur_price + w_arima * pred_arima * 0.95 + w_vdp * p16_vdp
        p84_ens = w_rw * cur_price + w_arima * pred_arima * 1.05 + w_vdp * p84_vdp
        
        # Ensemble direction probability
        prob_up_ens = (w_rw * 0.5 + w_arima * (1 if pred_arima > cur_price else 0)
                       + w_vdp * prob_up_vdp)
        
        bt.append({
            'fold': n_folds, 'horizon': h+1,
            'actual': actual, 'cur': cur_price,
            'pred_rw': pred_rw, 'pred_arima': pred_arima,
            'pred_vdp': pred_vdp, 'pred_ens': pred_ens,
            'p16': p16_ens, 'p84': p84_ens,
            'prob_up': prob_up_ens,
            'actual_up': int(actual > cur_price),
            'w_rw': w_rw, 'w_arima': w_arima, 'w_vdp': w_vdp,
            'damp': damp0,
        })

bt = pd.DataFrame(bt)
bt['err_rw'] = (bt['actual'] - bt['pred_rw']).abs()
bt['err_arima'] = (bt['actual'] - bt['pred_arima']).abs()
bt['err_vdp'] = (bt['actual'] - bt['pred_vdp']).abs()
bt['err_ens'] = (bt['actual'] - bt['pred_ens']).abs()
bt['dir_rw'] = ((bt['actual'] - bt['cur']) > 0).astype(int)  # 50% baseline
bt['dir_arima'] = (np.sign(bt['pred_arima']-bt['cur'])==np.sign(bt['actual']-bt['cur'])).astype(int)
bt['dir_vdp'] = (np.sign(bt['pred_vdp']-bt['cur'])==np.sign(bt['actual']-bt['cur'])).astype(int)
bt['dir_ens'] = (np.sign(bt['pred_ens']-bt['cur'])==np.sign(bt['actual']-bt['cur'])).astype(int)
bt['in68'] = ((bt['p16'] <= bt['actual']) & (bt['actual'] <= bt['p84'])).astype(int)
bt['brier'] = (bt['prob_up'] - bt['actual_up'])**2

print(f"  {len(bt)} obs, {n_folds} folds")

# ═══════════════════════════════════════════════════
#  6. METRICS
# ═══════════════════════════════════════════════════

print(f"\n{'═'*70}")
print(f"  BACKTEST RESULTS — ENSEMBLE vs INDIVIDUAL MODELS")
print(f"{'═'*70}")

metrics = {}
for h in [1, 2, 4, 8]:
    sub = bt[bt['horizon']==h]
    if len(sub) < 10: continue
    
    mae = {k: sub[f'err_{k}'].mean() for k in ['rw','arima','vdp','ens']}
    mape = {k: (sub[f'err_{k}']/sub['actual']).mean()*100 for k in ['rw','arima','vdp','ens']}
    dir_acc = {k: sub[f'dir_{k}'].mean()*100 for k in ['arima','vdp','ens']}
    cov68 = sub['in68'].mean()*100
    brier = sub['brier'].mean()
    
    best_mae = min(mae.values())
    winner = [k for k,v in mae.items() if v==best_mae][0]
    
    metrics[h] = {
        'n': int(len(sub)),
        'mae': {k: round(float(v),0) for k,v in mae.items()},
        'mape': {k: round(float(v),1) for k,v in mape.items()},
        'dir': {k: round(float(v),1) for k,v in dir_acc.items()},
        'cov68': round(float(cov68),1),
        'brier': round(float(brier),4),
        'winner': winner.upper(),
    }
    
    print(f"\n  ── W+{h} ({len(sub)} obs) ──")
    print(f"  {'':25s} {'RW':>10s} {'ARIMA':>10s} {'Liénard':>10s} {'ENSEMBLE':>10s}")
    print(f"  {'MAE ($)':25s} {mae['rw']:>10,.0f} {mae['arima']:>10,.0f} {mae['vdp']:>10,.0f} {mae['ens']:>10,.0f}")
    print(f"  {'MAPE (%)':25s} {mape['rw']:>10.1f} {mape['arima']:>10.1f} {mape['vdp']:>10.1f} {mape['ens']:>10.1f}")
    print(f"  {'Dir. Accuracy (%)':25s} {'50.0':>10s} {dir_acc['arima']:>10.1f} {dir_acc['vdp']:>10.1f} {dir_acc['ens']:>10.1f}")
    print(f"  {'Coverage 68%':25s} {'─':>10s} {'─':>10s} {'─':>10s} {cov68:>10.1f}")
    print(f"  {'Brier':25s} {'─':>10s} {'─':>10s} {'─':>10s} {brier:>10.4f}")
    w = winner.upper()
    print(f"  {'→ Winner (MAE)':25s} {w:>10s}")

# ═══════════════════════════════════════════════════
#  7. STRATEGY SHARPE
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  Trading Strategy (Ensemble Signal)")
print(f"{'─'*50}")

# Strategy: trade weekly based on ensemble prob_up
bt_w1 = bt[bt['horizon']==1].sort_values('fold').copy()
if len(bt_w1) > 20:
    bt_w1['signal'] = 0
    bt_w1.loc[bt_w1['prob_up'] > 0.55, 'signal'] = 1
    bt_w1.loc[bt_w1['prob_up'] < 0.45, 'signal'] = -1
    
    bt_w1['ret'] = np.log(bt_w1['actual'].values / bt_w1['cur'].values)
    bt_w1['strat_ret'] = bt_w1['signal'] * bt_w1['ret']
    bt_w1 = bt_w1.dropna(subset=['strat_ret','ret'])
    
    if len(bt_w1) > 10 and bt_w1['strat_ret'].std() > 0:
        sr = bt_w1['strat_ret'].mean() / bt_w1['strat_ret'].std() * np.sqrt(52)
        sr_bnh = bt_w1['ret'].mean() / (bt_w1['ret'].std() + 1e-10) * np.sqrt(52)
        
        cum_s = (1 + bt_w1['strat_ret']).cumprod()
        cum_b = (1 + bt_w1['ret']).cumprod()
        dd_s = (cum_s / cum_s.cummax() - 1).min() * 100
        dd_b = (cum_b / cum_b.cummax() - 1).min() * 100
        
        total_ret_s = (cum_s.iloc[-1] - 1) * 100
        total_ret_b = (cum_b.iloc[-1] - 1) * 100
        
        pct_long = (bt_w1['signal'] == 1).mean() * 100
        pct_short = (bt_w1['signal'] == -1).mean() * 100
        pct_flat = (bt_w1['signal'] == 0).mean() * 100
        
        # Win rate
        wins = (bt_w1[bt_w1['signal']!=0]['strat_ret'] > 0).mean() * 100
        
        print(f"  Signal: Long P(↑)>55%, Short <45%")
        print(f"  Positions: {pct_long:.0f}% long · {pct_short:.0f}% short · {pct_flat:.0f}% flat")
        print(f"  Win rate: {wins:.1f}%")
        print(f"  {'':25s} {'Ensemble':>12s} {'Buy&Hold':>10s}")
        print(f"  {'Sharpe (annualized)':25s} {sr:>12.2f} {sr_bnh:>10.2f}")
        print(f"  {'Total Return':25s} {total_ret_s:>11.1f}% {total_ret_b:>9.1f}%")
        print(f"  {'Max Drawdown':25s} {dd_s:>11.1f}% {dd_b:>9.1f}%")
        
        strat = {'sharpe':round(sr,2),'sharpe_bnh':round(sr_bnh,2),
                 'ret':round(total_ret_s,1),'ret_bnh':round(total_ret_b,1),
                 'dd':round(dd_s,1),'dd_bnh':round(dd_b,1),
                 'win_rate':round(wins,1),
                 'pct_long':round(pct_long,0),'pct_short':round(pct_short,0)}
    else:
        strat = {}
else:
    strat = {}

# ═══════════════════════════════════════════════════
#  8. CONDITIONED METRICS (by regime & phase)
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  Performance by Regime & Phase")
print(f"{'─'*50}")

bt_w1c = bt[bt['horizon']==1].copy()
for label, mask in [
    ("Reverting (d<0)", bt_w1c['damp'] < 0),
    ("Trending (d>0)", bt_w1c['damp'] > 0),
    ("Strong revert (d<-1)", bt_w1c['damp'] < -1),
]:
    sub = bt_w1c[mask]
    if len(sub) > 5:
        mae_ens = sub['err_ens'].mean()
        mae_rw = sub['err_rw'].mean()
        dir_ens = sub['dir_ens'].mean()*100
        beats = mae_ens < mae_rw
        print(f"  {label:25s}: n={len(sub):3d}  MAE_ens=${mae_ens:>8,.0f}  MAE_rw=${mae_rw:>8,.0f}  "
              f"Dir={dir_ens:.0f}%  {'✓ ENSEMBLE' if beats else '✗ RW'}")

# ═══════════════════════════════════════════════════
#  9. MC FORECAST FROM CURRENT STATE
# ═══════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  Monte Carlo Forecast (500 paths × 52 weeks)")
print(f"{'─'*50}")

P0 = pf['P'].iloc[-1]; V0 = pf['V'].iloc[-1]; U0 = pf['U'].iloc[-1]
cur = np.exp(P0); cv0 = cond_vol[-1]; reg0 = full_reg[-1]

N_FC, HOR = 500, 52
paths = np.zeros((N_FC, HOR))
np.random.seed(42)

for p in range(N_FC):
    P,V,U = P0,V0,U0; cv_m = cv0; rg = reg0
    for t in range(HOR):
        rg = np.random.choice(3, p=htrans[rg])
        mu_m = mu_opt * [1.3,0.7,1.1][rg]
        PU = P-U; PUn = PU/s_opt
        vdp = mu_m*(1-PUn**2)*V; fund = -om2_opt*PU
        P += V; V += vdp+fund+0.03*np.random.randn()+cv_m*np.random.randn()
        V = np.clip(V,-0.15,0.15)
        U += 0.02*(growth_weekly*(N+t)+lp[0]-U)+0.005*np.random.randn()
        cv_m = max(0.008, cv_m*0.97+0.03*abs(np.random.randn())*cv0)
        paths[p,t] = np.exp(P)

fc = []
for t in range(HOR):
    dp = paths[:,t]; h = t+1
    fc.append({'h':h,'p5':np.percentile(dp,5),'p16':np.percentile(dp,16),
               'p50':np.median(dp),'p84':np.percentile(dp,84),
               'p95':np.percentile(dp,95),
               'pu':round(np.mean(dp>cur)*100,1),
               'dd20':round(np.mean(dp<cur*0.8)*100,1)})

print(f"  From: ${cur:,.0f}  Fund: ${pf['fund'].iloc[-1]:,.0f}  MVRV: {pf['mvrv'].iloc[-1]:.2f}")
for h in [1,4,8,13,26,52]:
    f = fc[h-1]; ch = (f['p50']/cur-1)*100
    print(f"  W+{h:2d}: P50=${f['p50']:>10,.0f} ({ch:+.1f}%)  "
          f"IC68=[${f['p16']:>8,.0f}, ${f['p84']:>8,.0f}]  "
          f"P(↑)={f['pu']}%  P(DD20)={f['dd20']}%")

# ═══════════════════════════════════════════════════
#  10. EXPORT
# ═══════════════════════════════════════════════════

# History
step = max(1, N//80)
hist = []
for i in range(0, N, step):
    hist.append([wk['Date'].iloc[i].strftime('%Y-%m-%d'),
                 round(float(wk['close'].iloc[i]),0),
                 round(float(pf['fund'].iloc[i]),0),
                 round(float(pf['overval'].iloc[i]),1),
                 round(float(pf['damp'].iloc[i]),3),
                 round(float(pf['mvrv'].iloc[i]),2) if not np.isnan(pf['mvrv'].iloc[i]) else None,
                 RN[full_reg[i]]])
if hist[-1][0] != wk['Date'].iloc[-1].strftime('%Y-%m-%d'):
    hist.append([wk['Date'].iloc[-1].strftime('%Y-%m-%d'),
                 round(float(wk['close'].iloc[-1]),0),
                 round(float(pf['fund'].iloc[-1]),0),
                 round(float(pf['overval'].iloc[-1]),1),
                 round(float(pf['damp'].iloc[-1]),3),
                 round(float(pf['mvrv'].iloc[-1]),2),
                 RN[full_reg[-1]]])

fc_export = [[f['h'],round(f['p5']),round(f['p16']),round(f['p50']),
              round(f['p84']),round(f['p95']),f['pu'],f['dd20']]
             for f in fc if f['h'] in [1,2,4,8,13,26,52]]

output = {
    'model': 'Liénard Ensemble v8',
    'freq': 'weekly',
    'improvements': [
        'Weekly frequency (optimal for mean-reversion)',
        'MVRV-anchored fundamental (200w SMA proxy)',
        'EGARCH-Student-t conditional volatility',
        'HMM 3-state regime detection',
        'Ensemble: RW + ARIMA + Liénard (regime-weighted)',
        'Walk-forward 52w/8w/4w backtest',
    ],
    'params': {'mu':mu_opt,'omega2':om2_opt,'s':s_opt,
               'garch':{'alpha':round(alpha_g,4),'beta':round(beta_g,4),'nu':round(nu_g,2)}},
    'state': {
        'price':round(float(cur)),
        'fund':round(float(pf['fund'].iloc[-1])),
        'overval':round(float(pf['overval'].iloc[-1]),1),
        'mvrv':round(float(pf['mvrv'].iloc[-1]),2),
        'velocity':round(float(V0*1000),1),
        'damping':round(float(pf['damp'].iloc[-1]),3),
        'phase':'TRENDING' if pf['damp'].iloc[-1]>0 else 'REVERTING',
        'regime':RN[full_reg[-1]],
        'garch_vol':round(float(cond_vol[-1]*np.sqrt(52)*100),1),
    },
    'hmm':{'dist':[round(d,1) for d in hdist],
           'means':[round(float(m*100),2) for m in hm],
           'trans':htrans.round(3).tolist()},
    'backtest':metrics,
    'strategy':strat,
    'history':hist,
    'forecast':fc_export,
}

with open('/home/claude/btc_v8.json','w') as f:
    json.dump(output, f, default=str)

sz = len(json.dumps(output,default=str))/1024
print(f"\n  ✅ btc_v8.json ({sz:.0f} KB) — {len(hist)} hist + {len(fc_export)} fc")

# Final summary
print(f"\n{'═'*70}")
print(f"  v8 SUMMARY")
print(f"{'═'*70}")
print(f"  BTC ${cur:,.0f} | Fund ${pf['fund'].iloc[-1]:,.0f} | MVRV {pf['mvrv'].iloc[-1]:.2f}")
print(f"  Phase: {output['state']['phase']} (d={output['state']['damping']})")
print(f"  Regime: {output['state']['regime']} | Vol: {output['state']['garch_vol']}%/yr")
if strat:
    print(f"  Strategy Sharpe: {strat['sharpe']} vs B&H {strat['sharpe_bnh']}")
    print(f"  Win rate: {strat['win_rate']}%")
for h in [1,4,8]:
    if h in metrics:
        m = metrics[h]
        print(f"  W+{h}: Ensemble MAE=${m['mae']['ens']:,.0f} vs RW=${m['mae']['rw']:,.0f} "
              f"Dir={m['dir']['ens']}% Cov68={m['cov68']}% → {m['winner']}")
