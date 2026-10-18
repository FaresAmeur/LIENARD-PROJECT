"""Calcule et publie les verdicts à 26 semaines des entrées du registre R5.

Méthode figée par protocol_amendment_v2_3.json (déclarée avant son premier
usage, le 2026-11-27) : prix d'entrée = prix publié dans l'entrée ; prix de
sortie = Coin Metrics PriceUSD quotidien daté exactement eval_due ; paniers
équipondérés ; rendements bruts ; aucun seuil de réussite — un verdict est
un ensemble de chiffres, publié quel que soit le résultat.

Puis régénère les blocs « Verdicts » du README et de site/index.html, pour
que la vitrine ne puisse pas dériver des verdicts publiés.
"""
import json, hashlib, datetime, glob, os, re, sys, statistics, requests

CM_API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
PROTOCOL = "R5 evaluation — protocol amendment v2.3"
OVERDUE_DAYS = 7   # au-delà, un verdict toujours en attente est une vraie anomalie

REGISTRY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REGISTRY)


def sha256_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def entry_no(path):
    return int(re.search(r'entry(\d+)', os.path.basename(path)).group(1))


def fetch_price(asset, day):
    """PriceUSD daté exactement `day`, ou None s'il n'est pas (encore) publié."""
    resp = requests.get(CM_API, params={
        'assets': asset.lower(), 'metrics': 'PriceUSD', 'frequency': '1d',
        'start_time': day, 'end_time': day,
    }, timeout=30)
    resp.raise_for_status()
    for row in resp.json().get('data', []):
        if row['time'][:10] == day and row.get('PriceUSD') is not None:
            return float(row['PriceUSD'])
    return None


def mean_or_none(xs):
    return round(statistics.mean(xs), 6) if xs else None


def pp(a, b):
    return round((a - b) * 100, 2) if a is not None and b is not None else None


def build_verdict(entry_path, entry, due, prices):
    assets = []
    for r in entry['entries']:
        exit_price = prices[r['asset']]
        assets.append({
            'asset': r['asset'], 'signal_26w': r['signal_26w'],
            'entry_price': r['price'], 'exit_price': round(exit_price, 6),
            'return': round(exit_price / r['price'] - 1, 6),
        })
    longs = [a['return'] for a in assets if a['signal_26w'] == 'LONG']
    flats = [a['return'] for a in assets if a['signal_26w'] == 'FLAT']
    long_m, flat_m = mean_or_none(longs), mean_or_none(flats)
    univ_m = mean_or_none([a['return'] for a in assets])
    verdict = {
        'type': 'verdict',
        'protocol': PROTOCOL,
        'entry_file': os.path.basename(entry_path),
        'entry_sha256': entry['sha256'],
        'observation_date': entry['entries'][0]['date'],
        'eval_due': due,
        'exit_price_source': 'Coin Metrics Community API v4, PriceUSD, frequency=1d, time=eval_due',
        'evaluated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'assets': assets,
        'summary': {
            'n_assets': len(assets), 'n_long': len(longs), 'n_flat': len(flats),
            'long_mean': long_m, 'flat_mean': flat_m, 'universe_mean': univ_m,
            'long_minus_universe_pp': pp(long_m, univ_m),
            'long_minus_flat_pp': pp(long_m, flat_m),
            'long_hit_rate': round(sum(r > 0 for r in longs) / len(longs), 4) if longs else None,
        },
    }
    verdict['sha256'] = sha256_of(verdict)
    return verdict


def evaluate(registry_dir=REGISTRY, fetch=fetch_price, today=None):
    """Écrit les verdicts dus et disponibles. Renvoie (écrits, en retard)."""
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    cache, written, overdue = {}, [], []
    entries = sorted(glob.glob(os.path.join(registry_dir, 'r5_*entry*.json')), key=entry_no)
    for path in entries:
        n = entry_no(path)
        out = os.path.join(registry_dir, f'verdict_entry{n:03d}.json')
        if os.path.exists(out):
            continue
        entry = json.load(open(path, encoding='utf-8'))
        dues = {r['eval_due'] for r in entry['entries']}
        assert len(dues) == 1, f"{path}: eval_due incohérents {dues}"
        due = dues.pop()
        if datetime.date.fromisoformat(due) > today:
            continue
        prices = {}
        for r in entry['entries']:
            key = (r['asset'], due)
            if key not in cache:
                try:
                    cache[key] = fetch(r['asset'], due)
                except requests.RequestException as exc:
                    print(f"  ! {r['asset']} {due}: API indisponible ({exc}) — nouvel essai au prochain run")
                    cache[key] = None
            prices[r['asset']] = cache[key]
        if any(p is None for p in prices.values()):
            late = (today - datetime.date.fromisoformat(due)).days
            print(f"  … entrée {n:03d} (échéance {due}) : prix de sortie pas encore publié ({late} j)")
            if late > OVERDUE_DAYS:
                overdue.append(n)
            continue
        verdict = build_verdict(path, entry, due, prices)
        # newline='\n' : le verdict est horodaté octet par octet, il doit être identique sur tout OS.
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(verdict, f, indent=2)
        s = verdict['summary']
        print(f"  ✅ verdict entrée {n:03d} : LONG {s['long_mean']} | FLAT {s['flat_mean']} | univers {s['universe_mean']}")
        written.append(out)
    return written, overdue


# ── Vitrine ──────────────────────────────────────────────────────────────────

def _fr_date(iso):
    return datetime.date.fromisoformat(iso).strftime('%d/%m/%Y')


def _pct(x):
    return '—' if x is None else f"{x * 100:+.1f} %"


def _pts(x):
    return '—' if x is None else f"{x:+.1f} pts"


def load_state(registry_dir):
    entries = sorted(glob.glob(os.path.join(registry_dir, 'r5_*entry*.json')), key=entry_no)
    verdicts, upcoming = [], {}
    for path in entries:
        n = entry_no(path)
        vpath = os.path.join(registry_dir, f'verdict_entry{n:03d}.json')
        if os.path.exists(vpath):
            verdicts.append((n, json.load(open(vpath, encoding='utf-8'))))
        else:
            due = json.load(open(path, encoding='utf-8'))['entries'][0]['eval_due']
            upcoming.setdefault(due, []).append(n)
    return verdicts, sorted(upcoming.items())


def render_site(verdicts, upcoming):
    lines = ['<h2 id="verdicts">Verdicts à 26 semaines</h2>']
    if verdicts:
        lines.append('<div class="tbl"><table><tr><th>Entrée</th><th>Observation</th><th>LONG</th>'
                     '<th>FLAT</th><th>LONG − univers</th><th>LONG − FLAT</th></tr>')
        for n, v in verdicts:
            s = v['summary']
            lines.append(f'<tr><td>{n:03d}</td><td>{_fr_date(v["observation_date"])}</td>'
                         f'<td>{_pct(s["long_mean"])}</td><td>{_pct(s["flat_mean"])}</td>'
                         f'<td>{_pts(s["long_minus_universe_pp"])}</td><td>{_pts(s["long_minus_flat_pp"])}</td></tr>')
        lines.append('</table></div>')
    else:
        lines.append('<p>Aucun verdict pour l’instant. Chaque verdict est calculé automatiquement à son '
                     'échéance, publié qu’il soit bon ou mauvais, et ancré sur Bitcoin.</p>')
    if upcoming:
        nxt = '; '.join(f'{_fr_date(d)} (entrée{"s" if len(ns) > 1 else ""} '
                        f'{", ".join(f"{n:03d}" for n in ns)})' for d, ns in upcoming[:3])
        lines.append(f'<p>Prochaines échéances : {nxt}.</p>')
    lines.append('<p class="note">Méthode figée par l’amendement v2.3, déclaré avant le premier verdict : '
                 'prix publié dans l’entrée → prix Coin Metrics le jour de l’échéance, paniers équipondérés, '
                 'rendements bruts, aucun seuil de réussite. Les entrées 001 et 002 sont des sous-ensembles '
                 'de 003 (même observation du 29/05/2026).</p>')
    return '\n'.join(lines)


def render_readme(verdicts, upcoming):
    lines = []
    if verdicts:
        lines += ['| Entry | Observation | LONG | FLAT | LONG − universe | LONG − FLAT |',
                  '|---|---|---|---|---|---|']
        for n, v in verdicts:
            s = v['summary']
            lines.append(f'| {n:03d} | {v["observation_date"]} | {_pct(s["long_mean"])} | {_pct(s["flat_mean"])} '
                         f'| {_pts(s["long_minus_universe_pp"])} | {_pts(s["long_minus_flat_pp"])} |')
    else:
        lines.append('No verdict yet. Each one is computed automatically at its due date, '
                     'published whatever the outcome, and anchored in Bitcoin.')
    if upcoming:
        nxt = '; '.join(f'{d} (entr{"ies" if len(ns) > 1 else "y"} {", ".join(f"{n:03d}" for n in ns)})'
                        for d, ns in upcoming[:3])
        lines += ['', f'Next due: {nxt}.']
    lines += ['', 'Method fixed by protocol amendment v2.3 (`registry/protocol_amendment_v2_3.json`), '
                  'declared before the first verdict: published entry price → Coin Metrics price on the due '
                  'date, equal-weight baskets, gross returns, no success threshold. Entries 001 and 002 are '
                  'subsets of 003 (same observation).']
    return '\n'.join(lines)


def replace_block(text, body, path):
    # Le corps entre les marqueurs peut être vide (premier passage) : (?:.*?\n)? le tolère.
    pattern = r'(<!-- VERDICTS:START -->\n)(?:.*?\n)?(<!-- VERDICTS:END -->)'
    new, count = re.subn(pattern, lambda m: m.group(1) + body + '\n' + m.group(2), text, flags=re.S)
    assert count == 1, f"{path}: marqueurs VERDICTS introuvables — vitrine NON mise à jour."
    return new


def render_showcase(registry_dir=REGISTRY, root=ROOT):
    verdicts, upcoming = load_state(registry_dir)
    for rel, body in (('README.md', render_readme(verdicts, upcoming)),
                      ('site/index.html', render_site(verdicts, upcoming))):
        path = os.path.join(root, rel)
        text = open(path, encoding='utf-8').read()
        new = replace_block(text, body, path)
        if new != text:
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(new)
            print(f"  vitrine mise à jour : {rel}")


if __name__ == '__main__':
    written, overdue = evaluate()
    render_showcase()
    if overdue:
        # Code 3 (et non 1) : le workflow committe d'abord ce qui a été calculé,
        # puis passe le run au rouge — un retard est une vraie anomalie à voir.
        print(f"⛔ Verdicts en retard de plus de {OVERDUE_DAYS} jours : entrées {overdue}", file=sys.stderr)
        sys.exit(3)
