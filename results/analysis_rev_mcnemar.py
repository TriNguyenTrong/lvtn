# -*- coding: utf-8 -*-
"""McNemar chính xác hai phía (nhị thức) + Holm trong từng họ so sánh, cho các lần chạy lịch dài (25–29/9)
và CROHME 2023. Chạy từ results/:  python analysis_rev_mcnemar.py > analysis_rev_mcnemar.txt
Mẫu thiếu dòng trong file kết quả tính là sai. Mẫu số 986/1.147/1.199 (gộp 3.332) và 2.300 cho 2023."""
import math

YEARS = ("2014", "2016", "2019")


def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def load(prefix):
    d = {}
    for y in YEARS:
        for line in open(f"{prefix}_{y}_results.txt", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) == 4 and p[1] in ("CORRECT", "WRONG"):
                d[(y, p[0])] = p[1] == "CORRECT"
    return d


def load23(tag):
    d = {}
    for line in open(f"rev_crohme2023_{tag}_results.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] in ("CORRECT", "WRONG"):
            d[p[0]] = p[1] == "CORRECT"
    return d


def mcnemar(a, b):
    keys = set(a) | set(b)
    nb = sum(1 for k in keys if a.get(k, False) and not b.get(k, False))
    nc = sum(1 for k in keys if b.get(k, False) and not a.get(k, False))
    n, m = nb + nc, min(nb, nc)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return nb, nc, p


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def family(name, pairs, R, total):
    res = [(a, b) + mcnemar(R[a], R[b]) for a, b in pairs]
    adj = holm([r[4] for r in res])
    print(f"== {name}")
    for (a, b, nb, nc, p), h in zip(res, adj):
        ra = 100 * sum(R[a].values()) / total
        rb = 100 * sum(R[b].values()) / total
        print(f"  {a} ({ra:.2f}) vs {b} ({rb:.2f}): b={nb} c={nc} p={p:.3g} Holm={h:.3g}")


TAGS = ["long_offline", "offline_s13", "offline_s42", "long_shared", "shared_s13", "shared_s42",
        "cascaded_s7", "concat_s7", "abl_sares", "abl_sepcross", "abl_cascx"]
R = {t: load("rev_" + t) for t in TAGS}
family("shared-query vs offline-only, cung seed", [("long_shared", "long_offline"), ("shared_s13", "offline_s13"),
       ("shared_s42", "offline_s42")], R, 3332)
family("thiet ke hop nhat, seed 7", [("long_shared", "cascaded_s7"), ("long_shared", "concat_s7"),
       ("cascaded_s7", "concat_s7"), ("cascaded_s7", "long_offline"), ("concat_s7", "long_offline")], R, 3332)
family("ablation lop giai ma, seed 7", [("long_shared", "abl_sares"), ("long_shared", "abl_sepcross"),
       ("cascaded_s7", "abl_cascx")], R, 3332)
C = {t: load23(t) for t in ["long_shared", "shared_s13", "shared_s42", "long_offline", "offline_s13", "offline_s42"]}
family("CROHME 2023 (n=2300), shared-query vs offline-only, cung seed", [("long_shared", "long_offline"),
       ("shared_s13", "offline_s13"), ("shared_s42", "offline_s42")], C, 2300)


# Single comparisons reported without correction in Sections 4.7.2 and 4.7.3 (seed 7, main protocol)
G = {t: load("rev_" + t) for t in ["long_shared", "long_uni", "long_online", "long_offline", "long_trfenc"]}
print("== so sanh don, seed 7 (khong hieu chinh)")
for a, b in [("long_shared", "long_uni"), ("long_shared", "long_online"), ("long_trfenc", "long_shared"),
             ("long_trfenc", "long_offline"), ("long_trfenc", "long_online")]:
    nb, nc, p = mcnemar(G[a], G[b])
    ra = 100 * sum(G[a].values()) / 3332
    rb = 100 * sum(G[b].values()) / 3332
    print(f"  {a} ({ra:.2f}) vs {b} ({rb:.2f}): b={nb} c={nc} p={p:.3g}")
