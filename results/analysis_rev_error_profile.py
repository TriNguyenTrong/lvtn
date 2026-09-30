# Error analysis of Section 4.9 (main system under the main protocol, seed 7).
# Same procedure as analysis_traj_error_profile.py, applied to the long-schedule runs rev_*.
# Run from results/:
#   python3 analysis_rev_error_profile.py > analysis_rev_error_profile.txt
# Official denominators 986 / 1147 / 1199 = 3332. The rev_* files contain one line per test
# expression; an expression without output (505_em_51, over the 200-token limit) has an empty
# prediction and is counted as wrong. Token-level statistics skip it (1,484 of 1,485 errors).
from collections import Counter

YRS = ['2014', '2016', '2019']
DEN = {'2014': 986, '2016': 1147, '2019': 1199}
MAIN = 'rev_long_shared'
OFF = 'rev_long_offline'
CFGS = [MAIN, OFF]


def load(cfg, yr):
    d = {}
    for line in open(f'{cfg}_{yr}_results.txt', encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) == 4 and p[1] in ('CORRECT', 'WRONG'):
            d[p[0]] = (p[1] == 'CORRECT', p[2], p[3])
    assert len(d) == DEN[yr], (cfg, yr, len(d))
    return d


D = {c: {y: load(c, y) for y in YRS} for c in CFGS}
TOT = sum(DEN.values())


def lev(a, b):
    if a == b: return 0
    m, n = len(a), len(b); dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[n]


STRUCT = set('{ } ^ _'.split()) | {'\\frac', '\\sqrt', '\\begin{matrix}', '\\end{matrix}'}


def ops(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (a[i-1] != b[j-1]))
    o = []; i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            if a[i-1] != b[j-1]: o.append(('sub', a[i-1], b[j-1]))
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            o.append(('del', a[i-1], '')); i -= 1
        else:
            o.append(('ins', '', b[j-1])); j -= 1
    return o


for cfg in CFGS:
    ks = [sum(v[0] for v in D[cfg][y].values()) for y in YRS]
    print(f'{cfg}: correct {ks}, micro {100*sum(ks)/TOT:.2f}%, wrong {TOT-sum(ks)}')

print('\n== ExpRate, exact / <=1 / <=2 token edits (main system, micro over 3332) ==')
T = [0, 0, 0]
for y in YRS:
    for ok, pr, gt in D[MAIN][y].values():
        d = 0 if ok else lev(pr.split(), gt.split())
        for i in range(3):
            if d <= i: T[i] += 1
print(f'exact {100*T[0]/TOT:.2f} | <=1 {100*T[1]/TOT:.2f} | <=2 {100*T[2]/TOT:.2f}')

print('\n== Error classes (main system; expressions with an output) ==')
cls = {'structural': 0, 'symbol': 0, 'insdel': 0}; noout = []
for y in YRS:
    for k, (ok, pr, gt) in D[MAIN][y].items():
        if ok: continue
        if not pr.strip(): noout.append(k); continue
        o = ops(pr.split(), gt.split())
        if any(x[1] in STRUCT or x[2] in STRUCT for x in o): cls['structural'] += 1
        elif all(x[0] == 'sub' for x in o): cls['symbol'] += 1
        else: cls['insdel'] += 1
n = sum(cls.values())
print(f'errors with output: {n}; without output: {noout}')
for k, v in cls.items(): print(f'   {k:12s} {v:5d}  {100*v/n:5.1f}%')

cnt = Counter(); nsub = 0
for y in YRS:
    for ok, pr, gt in D[MAIN][y].values():
        if ok or not pr.strip(): continue
        for o in ops(pr.split(), gt.split()):
            if o[0] == 'sub' and o[1] not in STRUCT and o[2] not in STRUCT:
                nsub += 1; cnt[(o[2], o[1])] += 1
case = sum(v for (t, p), v in cnt.items() if t.lower() == p.lower() and t != p)
print(f'\n== Symbol substitutions (main system) == {nsub}; case confusions {case} ({100*case/nsub:.1f}%)')
for (t, p), v in cnt.most_common(10): print(f'   {t:10s} -> {p:10s} {v}')

print('\n== Error rate by target length (all 3332 expressions) ==')
for cfg in CFGS:
    by = {}
    for y in YRS:
        for ok, pr, gt in D[cfg][y].values():
            b = min((len(gt.split()) - 1) // 10, 3); t = by.setdefault(b, [0, 0]); t[1] += 1
            if not ok: t[0] += 1
    print(cfg, '  '.join(f'{lb}: {100*by[b][0]/by[b][1]:.1f}% ({by[b][0]}/{by[b][1]})'
                         for b, lb in enumerate(['1-10', '11-20', '21-30', '>30'])))

fix = broke = 0
for y in YRS:
    for k, v in D[MAIN][y].items():
        w = D[OFF][y][k]
        if v[0] and not w[0]: fix += 1
        if w[0] and not v[0]: broke += 1
print(f'\n== Main system vs offline-only (seed 7) == main only correct {fix}, offline only correct {broke}')
