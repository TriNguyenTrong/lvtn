# So sanh bon bo nhan dang quy dao->SRT duoi cung mot bo giai ma.
# Chay tu thu muc results/:
#   python analysis_srt_recognizers.py > analysis_srt_recognizers.txt
# McNemar chinh xac hai phia (nhi thuc), cung cong thuc voi
# analysis_significance_editdist.py de so lieu nhat quan voi Chuong 4.
import math
import os

if hasattr(math, "comb"):          # env bttr chay Python 3.7, chua co math.comb
    comb = math.comb
else:
    def comb(n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

CFGS = {
    "thay_v4":   "traj_abl_srtoof_thayv4full",
    "thay_v1":   "traj_abl_srtoof_thayckpt",
    "ta_4chieu": "traj_abl_srtoof",
    "ta_8chieu": "traj_abl_srtoof_better",
}
YRS = ["2014", "2016", "2019"]


def load(prefix, yr):
    path = f"{prefix}_{yr}_results.txt"
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 4 and p[1] in ("CORRECT", "WRONG"):
            d[p[0]] = (p[1] == "CORRECT")
    return d


D = {}
for name, prefix in CFGS.items():
    D[name] = {y: load(prefix, y) for y in YRS}
    missing = [y for y in YRS if D[name][y] is None]
    if missing:
        print(f"!! thieu ket qua {name}: {', '.join(missing)}")

print("== EXPRATE ==")
print(f"{'bo nhan dang':14s} " + "  ".join(f"{y:>16s}" for y in YRS) + "        micro")
for name in CFGS:
    cells, k_all, n_all = [], 0, 0
    for y in YRS:
        d = D[name][y]
        if d is None:
            cells.append(f"{'--':>16s}")
            continue
        k, n = sum(d.values()), len(d)
        k_all += k
        n_all += n
        cells.append(f"{100*k/n:6.2f}% ({k}/{n})")
    micro = f"{100*k_all/n_all:6.2f}%" if n_all else "   --"
    print(f"{name:14s} " + "  ".join(cells) + f"   {micro}")


def mcnemar(a, b, years):
    """b = a dung/b sai, c = nguoc lai; chi tinh tren mau co o ca hai."""
    nb = nc = 0
    for y in years:
        da, db = D[a][y], D[b][y]
        if da is None or db is None:
            continue
        for k in da:
            if k not in db:
                continue
            if da[k] and not db[k]:
                nb += 1
            elif db[k] and not da[k]:
                nc += 1
    n, m = nb + nc, min(nb, nc)
    if n == 0:
        return nb, nc, 1.0
    p = min(2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n, 1.0)
    return nb, nc, p


PAIRS = [
    ("thay_v4", "thay_v1"),
    ("thay_v4", "ta_4chieu"),
    ("thay_v4", "ta_8chieu"),
    ("ta_8chieu", "ta_4chieu"),
]

print("\n== McNEMAR chinh xac hai phia ==")
print("(b = cot trai dung / cot phai sai;  c = nguoc lai)")
for a, b in PAIRS:
    row = [f"{a} vs {b}:"]
    for y in YRS:
        nb, nc, p = mcnemar(a, b, [y])
        row.append(f"{y} b={nb} c={nc} p={p:.4f}")
    nb, nc, p = mcnemar(a, b, YRS)
    row.append(f"| gop b={nb} c={nc} p={p:.4g}")
    print("  " + "  ".join(row))
