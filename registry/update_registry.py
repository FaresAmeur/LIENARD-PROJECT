"""Génère l'entrée hebdo du registre — règle figée R5 v2.1 (z<-0.5 → LONG).

Source de données : Coin Metrics Community API v4 (protocol_amendment_v2_2.json,
2026-07-18) — remplace le CSV GitHub coinmetrics/data, figé depuis 2026-05-23.
"""
import pandas as pd, numpy as np, json, hashlib, datetime, glob, os, re, requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ASSETS = ['btc','eth','ltc','xrp','ada','doge']
CM_API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

# 1. Trouver la dernière entrée (pour chaîner) — tri par numéro d'entrée, pas alphabétique
entries = sorted(glob.glob('r5_*entry*.json'), key=lambda f: int(re.search(r'entry(\d+)', f).group(1)))
prev = json.load(open(entries[-1]))
n_next = int(re.search(r'entry(\d+)', entries[-1]).group(1)) + 1

# 2. Télécharger les données Coin Metrics (API REST) et calculer les signaux
rows = []
for a in ASSETS:
    resp = requests.get(CM_API, params={
        'assets': a, 'metrics': 'CapMVRVCur,PriceUSD', 'frequency': '1d',
        'page_size': 10000, 'sort': 'time', 'paging_from': 'start',
    }, timeout=30)
    resp.raise_for_status()
    d = pd.DataFrame(resp.json()['data'])
    d = d.dropna(subset=['CapMVRVCur', 'PriceUSD'])
    d['time'] = pd.to_datetime(d['time']).dt.tz_localize(None)
    d['CapMVRVCur'] = d['CapMVRVCur'].astype(float)
    d['PriceUSD'] = d['PriceUSD'].astype(float)
    w = d.set_index('time').resample('W-FRI').last().dropna()
    # Drop any trailing bin whose Friday label is still in the future: on a
    # non-Friday run, resample('W-FRI') labels the current partial week with
    # the upcoming Friday, which would emit a future-dated, incomplete entry.
    today = pd.Timestamp(datetime.datetime.now(datetime.timezone.utc).date())
    w = w[w.index <= today]
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

# 4. Régénérer le bloc signaux du README et de site/index.html à partir de
#    `rows`, pour que la vitrine ne puisse plus jamais dériver du registre.
longs = [r['asset'] for r in rows if r['signal_26w'] == 'LONG']
flats = [r['asset'] for r in rows if r['signal_26w'] == 'FLAT']
entry_date = rows[0]['date']
eval_due = rows[0]['eval_due']

readme_path = '../README.md'
readme = open(readme_path, encoding='utf-8').read()
track_record = (
    "## Track record (prospective, tamper-evident)\n"
    "`registry/` — hash-chained entries (each embeds SHA-256 of predecessor), OpenTimestamps\n"
    f"automation anchors every entry in Bitcoin. Current (entry {n_next:03d}, {entry_date}): "
    f"{', '.join(longs)} **LONG**; {', '.join(flats)} **FLAT**.\n"
    f"Evaluation due {eval_due}. Rule frozen: z(MVRV) < −0.5 → LONG.\n"
)
readme_new = re.sub(
    r"## Track record \(prospective, tamper-evident\)\n.*?\n(?=\n## )",
    track_record, readme, count=1, flags=re.S,
)
assert readme_new != readme, "README anchor '## Track record (prospective, tamper-evident)' not found — showcase NOT updated, fix the regex."
open(readme_path, 'w', encoding='utf-8').write(readme_new)

site_path = '../site/index.html'
site = open(site_path, encoding='utf-8').read()
d = datetime.date.fromisoformat(entry_date)
ed = datetime.date.fromisoformat(eval_due)
spans = ' '.join(
    f'<span class="sig {"long" if r["signal_26w"] == "LONG" else "flat"}">{r["asset"]} {r["signal_26w"]}</span>'
    for r in rows
)
signaux_line = (
    f'<h2>Signaux courants (entrée {n_next:03d}, {d:%d/%m/%Y}, éval. {ed:%d/%m/%Y})</h2>'
    f'<p>{spans}</p>'
)
site_new = re.sub(r'<h2>Signaux courants.*?</h2><p>.*?</p>', signaux_line, site, count=1, flags=re.S)
assert site_new != site, "index.html anchor '<h2>Signaux courants...' not found — showcase NOT updated, fix the regex."
open(site_path, 'w', encoding='utf-8').write(site_new)

print(f"✅ Vitrine régénérée : README.md + site/index.html (entrée {n_next:03d})")
