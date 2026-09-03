"""How much of the two-stage gap is the recogniser's fault?

Splits the test sets by whether the predicted SRT is exactly right, then reads
the expression accuracy inside each group. That turns "a better recogniser
would help" from a hope into an arithmetic: ExpRate = p*A + (1-p)*B, where p is
the share of expressions whose SRT comes out exactly right.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YRS = ["2014", "2016", "2019"]
PAIRS = [
    ("thay v4",    "srt_pred_thay_v4_full", "traj_abl_srtoof_thayv4full"),
    ("thay v1",    "srt_pred_thay",         "traj_abl_srtoof_thayckpt"),
    ("ta 4 chieu", "srt_pred_oof",          "traj_abl_srtoof"),
    ("ta 8 chieu", "srt_pred_oof_tap8n",    "traj_abl_srtoof_better"),
]


def load_gt():
    d = {}
    for line in open(os.path.join(ROOT, "crohme_all.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            d[os.path.splitext(os.path.basename(p[0]))[0]] = p[1].split()
    return d


gt = load_gt()
print(f"{'bo nhan dang':12s} {'p(SRT dung)':>12s} {'ExpRate|SRT dung':>18s} "
      f"{'ExpRate|SRT sai':>17s} {'ExpRate chung':>14s}")
rows = []
for name, pdir, rtag in PAIRS:
    ok_ok = ok_n = bad_ok = bad_n = 0
    for y in YRS:
        pp = os.path.join(ROOT, "online", pdir, f"{y}.txt")
        rp = os.path.join(ROOT, "results", f"{rtag}_{y}_results.txt")
        if not (os.path.exists(pp) and os.path.exists(rp)):
            continue
        pred = {}
        for line in open(pp, encoding="utf-8"):
            q = line.rstrip("\n").split("\t")
            pred[os.path.splitext(os.path.basename(q[0]))[0]] = q[1].split() if len(q) > 1 else []
        for line in open(rp, encoding="utf-8"):
            q = line.rstrip("\n").split("\t")
            if len(q) != 4 or q[1] not in ("CORRECT", "WRONG"):
                continue
            base, good = q[0], q[1] == "CORRECT"
            if base not in pred or base not in gt:
                continue
            if pred[base] == gt[base]:
                ok_n += 1
                ok_ok += good
            else:
                bad_n += 1
                bad_ok += good
    n = ok_n + bad_n
    p = ok_n / n
    A = 100 * ok_ok / max(ok_n, 1)
    B = 100 * bad_ok / max(bad_n, 1)
    tot = 100 * (ok_ok + bad_ok) / n
    rows.append((name, p, A, B, tot))
    print(f"{name:12s} {100*p:11.1f}% {A:17.2f}% {B:16.2f}% {tot:13.2f}%")

print("\n== SUY RA: can bao nhieu chuoi dung tron ven de dat muc luan van? ==")
A = sum(r[2] for r in rows) / len(rows)
B = sum(r[3] for r in rows) / len(rows)
print(f"  trung binh 4 bo: ExpRate|SRT dung = {A:.2f}%   ExpRate|SRT sai = {B:.2f}%")
for target, label in ((47.91, "bang nhanh chi anh"), (50.89, "bang he chinh 8 chieu (dual_shared)")):
    need = (target - B) / (A - B)
    print(f"  de dat {target:5.2f}% ({label}): can p = {100*need:5.1f}% "
          f"(hien cao nhat la {100*max(r[1] for r in rows):.1f}%)")
