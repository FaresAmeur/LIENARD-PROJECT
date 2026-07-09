"""Génère l'entrée hebdo du registre — règle figée R5 v2.1 (z<-0.5 → LONG)."""
import pandas as pd, numpy as np, json, hashlib, datetime, glob, os, re, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ASSETS = ['btc','eth','ltc','xrp','ada','doge']

# 1. Trouver la dernière entrée (pour chaîner) — tri par numéro d'entrée, pas alphabétique
entries = sorted(glob.glob('r5_*entry*.json'), key=lambda f: int(re.search(r'entry(\d+)', f).group(1)))
prev = json.load(open(entries[-1]))
n_next = int(re.search(r'entry(\d+)', entries[-1]).group(1)) + 1

# 2. Télécharger les données Coin Metrics et calculer les signaux
rows = []
for a in ASSETS:
    url = f"https://raw.githubusercontent.com/coinmetrics/data/master/csv/{a}.csv"
    d = pd.read_csv(url, usecols=['time','CapMVRVCur','PriceUSD']).dropna()
    d['time'] = pd.to_datetime(d['time'])
    w = d.set_index('time').resample('W-FRI').last().dropna()
    mv = w['CapMVRVCur'].values
    z = (mv[-1]-mv.mean())/mv.std()
    rows.append({'asset':a.upper(),'date':str(w.index[-1].date()),
        'price':round(float(w['PriceUSD'].iloc[-1]),4),'mvrv':round(float(mv[-1]),3),
        'z':round(float(z),2),'signal_26w':'LONG' if z<-0.5 else 'FLAT',
        'eval_due':str((w.index[-1]+pd.Timedelta(weeks=26)).date())})
    print(f"  {a.upper():5s} z={z:+.2f} → {rows[-1]['signal_26w']}")

# 2b. Garde-fou de fraîcheur — refuse un doublon si la source n'a pas bougé
if rows[0]['date'] == prev['entries'][0]['date']:
    raise SystemExit(f"⛔ Source stale: dernière donnée = {rows[0]['date']}, "
                     f"identique à l'entrée précédente. Aucune entrée créée.")

# 3. Chaîner, hasher, écrire
log = {'protocol':'R5 v2.1 — rule frozen (z<-0.5→LONG)',
       'prev_entry_sha256': prev['sha256'],
       'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'entries': rows}
log['sha256'] = hashlib.sha256(json.dumps(log,sort_keys=True).encode()).hexdigest()
fname = f'r5_log_entry{n_next:03d}.json'
json.dump(log, open(fname,'w'), indent=2)
print(f"\n✅ {fname}  chaîné à {prev['sha256'][:12]}")
print(f"Ensuite: ots stamp {fname} ; git add/commit/push")
