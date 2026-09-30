# Bang chung cho Muc 4.9 moi (error analysis he chinh: online = stroke, offline = image)
# Chay tu thu muc results/:
#   python3 analysis_traj_error_profile.py > analysis_traj_error_profile.txt
# Mau so chinh thuc: 986 / 1147 / 1199 = 3332. Mau 505_em_51 (CROHME 2014) vuot gioi han
# 200 token nen khong co du doan trong file ket qua -> tinh la SAI, cong vao mau so.
import math
from collections import Counter

YRS = ['2014', '2016', '2019']
DEN = {'2014': 986, '2016': 1147, '2019': 1199}      # mau so chinh thuc
MAIN = 'traj_abl_dual_shared_aux'
OFF  = 'traj_abl_offline_lrmax'
ON   = 'traj_abl_online_aux'
CFGS = [OFF, ON, 'traj_abl_concat_aux', 'traj_abl_cascaded_aux', MAIN, 'traj_abl_dual_shared_uni_aux']

def load(cfg, yr):
    d = {}
    for line in open(f'{cfg}_{yr}_results.txt', encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) == 4 and p[1] in ('CORRECT', 'WRONG'):
            d[p[0]] = (p[1] == 'CORRECT', p[2], p[3])
    return d

D = {c: {y: load(c, y) for y in YRS} for c in CFGS}
TOT = sum(DEN.values())

print('== BANG 6 (kiem chung ExpRate voi mau so chinh thuc) ==')
for c in CFGS:
    ks = [sum(v[0] for v in D[c][y].values()) for y in YRS]
    rates = [100 * k / DEN[y] for k, y in zip(ks, YRS)]
    print(f'{c:30s}', ' '.join(f'{r:5.2f}' for r in rates), f'micro={100*sum(ks)/TOT:5.2f}')

def lev(a, b):
    if a == b: return 0
    m, n = len(a), len(b); dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[n]

print('\n== ExpRate <=1 / <=2 (he chinh) ==')
T = [0, 0, 0]
for y in YRS:
    c = [0, 0, 0]
    for ok, pr, gt in D[MAIN][y].values():
        d = 0 if ok else lev(pr.split(), gt.split())
        for i, t in enumerate((0, 1, 2)):
            if d <= t: c[i] += 1
    print(f'{y}: exact {100*c[0]/DEN[y]:.2f} | <=1 {100*c[1]/DEN[y]:.2f} | <=2 {100*c[2]/DEN[y]:.2f}')
    for i in range(3): T[i] += c[i]
print(f'micro (n={TOT}): exact {100*T[0]/TOT:.2f} | <=1 {100*T[1]/TOT:.2f} | <=2 {100*T[2]/TOT:.2f}')

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

def profile(cfg):
    cls = {'structural': 0, 'symbol': 0, 'insdel': 0}; bylen = {}; wrong = 0
    for y in YRS:
        for ok, pr, gt in D[cfg][y].values():
            L = len(gt.split()); b = min((L - 1) // 10, 3)
            t = bylen.setdefault(b, [0, 0]); t[1] += 1
            if ok: continue
            wrong += 1; t[0] += 1
            o = ops(pr.split(), gt.split())
            if any(x[1] in STRUCT or x[2] in STRUCT for x in o): cls['structural'] += 1
            elif all(x[0] == 'sub' for x in o): cls['symbol'] += 1
            else: cls['insdel'] += 1
    return cls, bylen, wrong

for cfg in (MAIN, OFF, ON):
    cls, bylen, wrong = profile(cfg)
    print(f'\n== PHAN LOAI LOI [{cfg}] (tren {wrong} cau co dau ra) ==')
    for k, v in cls.items(): print(f'   {k:12s} {v:5d}  {100*v/wrong:5.1f}%')
    if cfg == MAIN:
        # 505_em_51 khong co dau ra -> chi cong vao ty le loi theo do dai (nhom >30)
        bylen[3][0] += 1; bylen[3][1] += 1
        print('   ty le loi theo do dai (da tinh ca 505_em_51 vao nhom >30):')
        for b, lb in enumerate(['1-10', '11-20', '21-30', '>30']):
            if b in bylen:
                x, n = bylen[b]; print(f'     {lb:6s}: {100*x/n:5.1f}% ({x}/{n})')

cnt = Counter(); nsub = 0; nedit = 0
for y in YRS:
    for ok, pr, gt in D[MAIN][y].values():
        if ok: continue
        for o in ops(pr.split(), gt.split()):
            nedit += 1
            if o[0] == 'sub' and o[1] not in STRUCT and o[2] not in STRUCT:
                nsub += 1; cnt[(o[2], o[1])] += 1
case = sum(v for (t, p), v in cnt.items() if t.lower() == p.lower() and t != p)
print(f'\n== THAY THE KY HIEU (he chinh) == tong edit={nedit}, thay the ky hieu thuan={nsub}')
print(f'   nham hoa-thuong: {case} ({100*case/nsub:.1f}% cua {nsub})')
for (t, p), v in cnt.most_common(10): print(f'   {t:10s} -> {p:10s} {v}')

fix = broke = 0
for y in YRS:
    for k, v in D[MAIN][y].items():
        if v[0] and not D[OFF][y][k][0]: fix += 1
        if D[OFF][y][k][0] and not v[0]: broke += 1
print(f'\n== FUSION vs OFFLINE-ONLY == sua duoc {fix}, lam hong {broke}, rong {fix-broke}')

def mcnemar(a, b, years=YRS):
    x = c = 0
    for y in years:
        for k, v in D[a][y].items():
            w = D[b][y][k]
            if v[0] and not w[0]: x += 1
            elif w[0] and not v[0]: c += 1
    n = x + c; m = min(x, c)
    p = min(2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n, 1.0) if n else 1.0
    return x, c, p

print('\n== McNEMAR exact 2-sided ==')
for a, b in [(MAIN, OFF), (MAIN, 'traj_abl_dual_shared_uni_aux'), (MAIN, 'traj_abl_cascaded_aux'),
             (MAIN, 'traj_abl_concat_aux'), ('traj_abl_cascaded_aux', 'traj_abl_concat_aux'), (MAIN, ON)]:
    x, c, p = mcnemar(a, b); print(f'   {a} vs {b}: b={x} c={c} p={p:.4g}')
for y in YRS:
    x, c, p = mcnemar(MAIN, OFF, [y]); print(f'   main vs offline {y}: b={x} c={c} p={p:.4g}')
for y in YRS:
    x, c, p = mcnemar(MAIN, 'traj_abl_dual_shared_uni_aux', [y]); print(f'   main vs uni {y}: b={x} c={c} p={p:.4g}')
