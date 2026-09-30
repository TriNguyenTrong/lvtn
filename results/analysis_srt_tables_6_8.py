# -*- coding: utf-8 -*-
"""Tính lại Bảng 6 và Bảng 8 (Mục 4.7.1) từ dữ liệu gốc — chỉ đọc, chạy CPU.
Chạy từ gốc repo:  python results/analysis_srt_tables_6_8.py > results/analysis_srt_tables_6_8.txt

Bảng 6: tỉ lệ lỗi token (edit distance / số token chuẩn) và tỉ lệ chuỗi đúng trọn vẹn trên CROHME 2019,
        so SRT dự đoán (online/srt_pred_*/2019.txt) với SRT chuẩn (crohme_all.txt).
Bảng 8: gộp 3 tập (986 + 1.147 + 1.199 = 3.332), tách theo SRT đúng/sai, ExpRate mỗi nhóm lấy từ
        file kết quả của Bảng 7 (results/traj_abl_srtoof*_{năm}_results.txt; mẫu thiếu dòng = sai).
"""
import os

N = {"2014": 986, "2016": 1147, "2019": 1199}
RECOGNIZERS = [  # (hàng trong bảng, thư mục SRT dự đoán, tiền tố file kết quả Bảng 7)
    ("Four-dimensional, initial version", "srt_pred_thay", "traj_abl_srtoof_thayckpt"),
    ("Four-dimensional, improved version", "srt_pred_thay_v4_full", "traj_abl_srtoof_thayv4full"),
    ("Four-dimensional, retrained", "srt_pred_oof", "traj_abl_srtoof"),
    ("Eight-dimensional, normalized", "srt_pred_tap8n", "traj_abl_srtoof_better"),
]


def base(p):
    return os.path.splitext(os.path.basename(p))[0]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


gt = {}
for line in open("crohme_all.txt", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        gt[base(p[0])] = p[1].split()

print("Bang 6 (CROHME 2019)                      TER      exact")
rows8 = []
for name, d, res in RECOGNIZERS:
    n_ex = c_n = c_ok = w_ok = 0
    for y in N:
        pred = {}
        for line in open(f"online/{d}/{y}.txt", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            pred[base(p[0])] = p[1].split() if len(p) == 2 else []
        st = {}
        for line in open(f"results/{res}_{y}_results.txt", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) == 4 and p[1] in ("CORRECT", "WRONG"):
                st[p[0]] = p[1] == "CORRECT"
        keys = {k for k in set(st) | set(pred) if k in gt}
        assert len(keys) == N[y], (name, y, len(keys))
        if y == "2019":
            err = sum(lev(pred.get(k, []), gt[k]) for k in keys)
            tok = sum(len(gt[k]) for k in keys)
            ex = sum(pred.get(k, []) == gt[k] for k in keys)
            print(f"{name:40s} {100*err/tok:6.2f}%  {100*ex/N[y]:5.1f}% ({ex}/{N[y]})")
        for k in keys:
            ok = st.get(k, False)
            if pred.get(k, []) == gt[k]:
                n_ex += 1; c_n += 1; c_ok += ok
            else:
                w_ok += ok
    tot = sum(N.values())
    rows8.append(f"{name:40s} {100*n_ex/tot:5.1f}   {100*c_ok/c_n:6.2f} ({c_ok}/{c_n})   "
                 f"{100*w_ok/(tot-c_n):6.2f} ({w_ok}/{tot-c_n})")
print("\nBang 8 (gop 3 tap, n=3.332)             exact SRT  ExpRate|SRT dung   ExpRate|SRT sai")
print("\n".join(rows8))
