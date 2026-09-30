# -*- coding: utf-8 -*-
"""Tong hop bang ablation SRT du doan tu net but + kiem dinh McNemar."""
import os
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
YEARS = ["2014", "2016", "2019"]
N_OFFICIAL = {"2014": 986, "2016": 1147, "2019": 1199}

ROWS = [
    ("offline-only", "traj_abl_offline_lrmax_{y}_results.txt"),
    ("online-only",  "traj_abl_srtoof_thayv4_online_{y}_results.txt"),
    ("concat",       "traj_abl_srtoof_thayv4_concat_{y}_results.txt"),
    ("cascaded",     "traj_abl_srtoof_thayv4_cascaded_{y}_results.txt"),
    ("shared-query", "traj_abl_srtoof_thayv4full_{y}_results.txt"),
]


def load(path):
    """Doc file ket qua, tra ve (total, correct, exprate, dict image->is_correct)."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[1].strip()
    # "Total: 985, Correct: 492, ExpRate: 49.95%"
    parts = header.replace(",", "").split()
    total = int(parts[1])
    correct = int(parts[3])
    exprate = float(parts[5].rstrip("%"))
    per_sample = {}
    for line in lines[4:]:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        img, status = cols[0], cols[1]
        per_sample[img] = (status.strip() == "CORRECT")
    return total, correct, exprate, per_sample


def mcnemar_exact_p(b, c):
    """Kiem dinh McNemar chinh xac hai phia (nhi thuc), b,c la so cap lech nhau."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * p_tail)
    return p


def micro_avg(vals_by_year, weights):
    num = sum(vals_by_year[y] * weights[y] for y in YEARS)
    den = sum(weights[y] for y in YEARS)
    return num / den


def main():
    out_lines = []
    data = {}  # row_name -> year -> (total, correct, exprate, per_sample)
    for name, tmpl in ROWS:
        data[name] = {}
        for y in YEARS:
            path = os.path.join(RESULTS, tmpl.format(y=y))
            if not os.path.exists(path):
                out_lines.append(f"!! THIEU FILE: {path}")
                continue
            data[name][y] = load(path)

    out_lines.append("BANG ABLATION - SRT DU DOAN TU NET BUT (khong phai SRT chuan)")
    out_lines.append("=" * 70)
    header = f"{'Thiet ke':<14}" + "".join(f"{y:>12}" for y in YEARS) + f"{'Micro-avg':>14}"
    out_lines.append(header)
    out_lines.append("-" * len(header))

    exprate_correct = {}  # name -> year -> correct count (for micro avg + McNemar)
    exprate_total = {}
    for name, _ in ROWS:
        exprate_correct[name] = {}
        exprate_total[name] = {}
        row_vals = []
        for y in YEARS:
            if y not in data[name]:
                row_vals.append("N/A")
                continue
            total, correct, exprate, _ = data[name][y]
            exprate_correct[name][y] = correct
            exprate_total[name][y] = total
            row_vals.append(f"{exprate:.2f}%")
        if all(y in data[name] for y in YEARS):
            micro = 100.0 * sum(exprate_correct[name][y] for y in YEARS) / sum(N_OFFICIAL[y] for y in YEARS)
            micro_str = f"{micro:.2f}%"
        else:
            micro_str = "N/A"
        out_lines.append(f"{name:<14}" + "".join(f"{v:>12}" for v in row_vals) + f"{micro_str:>14}")

    out_lines.append("")
    out_lines.append(f"(mau so chinh thuc dung cho micro-avg: 2014={N_OFFICIAL['2014']}, "
                      f"2016={N_OFFICIAL['2016']}, 2019={N_OFFICIAL['2019']}; "
                      f"tong = {sum(N_OFFICIAL.values())})")
    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append("KIEM DINH MCNEMAR CHINH XAC HAI PHIA (gop ca 3 nam, ghep cap theo anh)")
    out_lines.append("=" * 70)

    pairs = [
        ("concat", "offline-only", "fusion (concat) vs offline-only"),
        ("cascaded", "offline-only", "fusion (cascaded) vs offline-only"),
        ("shared-query", "offline-only", "fusion (shared-query) vs offline-only"),
        ("concat", "online-only", "fusion (concat) vs online-only"),
        ("cascaded", "online-only", "fusion (cascaded) vs online-only"),
        ("shared-query", "online-only", "fusion (shared-query) vs online-only"),
        ("concat", "cascaded", "concat vs cascaded"),
        ("concat", "shared-query", "concat vs shared-query"),
        ("cascaded", "shared-query", "cascaded vs shared-query"),
    ]

    for name_a, name_b, label in pairs:
        b_total = 0  # A dung, B sai
        c_total = 0  # A sai, B dung
        n_common = 0
        missing_years = []
        for y in YEARS:
            if y not in data[name_a] or y not in data[name_b]:
                missing_years.append(y)
                continue
            _, _, _, sa = data[name_a][y]
            _, _, _, sb = data[name_b][y]
            common_imgs = set(sa.keys()) & set(sb.keys())
            for img in common_imgs:
                n_common += 1
                ca, cb = sa[img], sb[img]
                if ca and not cb:
                    b_total += 1
                elif (not ca) and cb:
                    c_total += 1
        if missing_years:
            out_lines.append(f"{label}: THIEU nam {missing_years}, bo qua")
            continue
        p = mcnemar_exact_p(b_total, c_total)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        out_lines.append(
            f"{label:<40}  n={n_common:4d}  b(A dung,B sai)={b_total:4d}  "
            f"c(A sai,B dung)={c_total:4d}  p={p:.4g}  {sig}"
        )

    out_lines.append("")
    out_lines.append("Ghi chu: A = thiet ke dau tien trong ten cap so sanh, B = thiet ke thu hai.")
    out_lines.append("b = so mau A dung nhung B sai; c = so mau A sai nhung B dung.")
    out_lines.append("*** p<0.001, ** p<0.01, * p<0.05, n.s. = khong co y nghia thong ke (p>=0.05).")
    out_lines.append("")
    out_lines.append("Caveat: nhanh online dung SRT DU DOAN (khong phai oracle), huan luyen tren")
    out_lines.append("online/srt_pred_oof (out-of-fold, TER train 7.56%), danh gia bang bo nhan dien")
    out_lines.append("cua thay (online/srt_pred_thay_v4_full) tren ca 3 tap test.")

    out_path = os.path.join(RESULTS, "srtpred_ablation_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Da ghi: {out_path}")


if __name__ == "__main__":
    main()
