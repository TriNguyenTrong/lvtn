"""Where does a predicted SRT sequence differ from the ground-truth one?

The two-stage variant lives or dies on this: a single wrong token in the SRT
stream propagates into the expression the decoder produces. This script aligns
each prediction with its reference, classifies every edit as touching a symbol
or one of the seven spatial relations, and prints the confusions that dominate.

    python tools/srt_error_profile.py --pred-dir online/srt_pred_thay_v4_full \
        --split 2019 --examples 8 --out ghi_chu/srt_loi_thay_v4_2019.txt
"""
import argparse
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gt(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            out[os.path.splitext(os.path.basename(p[0]))[0]] = p[1].split()
    return out


def load_pred(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        out[os.path.splitext(os.path.basename(p[0]))[0]] = p[1].split() if len(p) > 1 else []
    return out


def align(a, b):
    """Edit script turning prediction `a` into reference `b`."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("keep" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("ins", a[i - 1], ""))      # prediction has a token too many
            i -= 1
        else:
            ops.append(("del", "", b[j - 1]))      # prediction dropped a token
            j -= 1
    return ops[::-1], dp[m][n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", default="online/srt_pred_thay_v4_full")
    p.add_argument("--split", default="2019")
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--vocab", default=os.path.join("vocab", "crohme_seq_vocab.txt"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    def full(x):
        return x if os.path.isabs(x) else os.path.join(ROOT, x)

    words = [w.strip() for w in open(full(args.vocab), encoding="utf-8") if w.strip()]
    rels = set(words[101:108])          # the notebook's own convention: 101..107

    gt = load_gt(os.path.join(ROOT, "crohme_all.txt"))
    pred = load_pred(os.path.join(full(args.pred_dir), f"{args.split}.txt"))
    keys = sorted(k for k in pred if k in gt)

    lines = []
    def w(s=""):
        lines.append(s)

    kinds = Counter()
    conf = Counter()
    rel_conf = Counter()
    n_err_seq = only_rel = only_sym = both = 0
    tot_ed = tot_ref = 0
    per_seq = []

    for k in keys:
        ops, ed = align(pred[k], gt[k])
        tot_ed += ed
        tot_ref += len(gt[k])
        bad = [o for o in ops if o[0] != "keep"]
        per_seq.append((ed, k, ops))
        if not bad:
            continue
        n_err_seq += 1
        touch_rel = touch_sym = False
        for kind, a, b in bad:
            tok = a or b
            is_rel = (a in rels) or (b in rels)
            kinds[(kind, "quan he" if is_rel else "ky hieu")] += 1
            if is_rel:
                touch_rel = True
            else:
                touch_sym = True
            if kind == "sub":
                (rel_conf if is_rel else conf)[(a, b)] += 1
            elif kind == "ins":
                (rel_conf if is_rel else conf)[(a, "(thua)")] += 1
            else:
                (rel_conf if is_rel else conf)[("(thieu)", b)] += 1
        if touch_rel and touch_sym:
            both += 1
        elif touch_rel:
            only_rel += 1
        else:
            only_sym += 1

    n = len(keys)
    w(f"BO SRT: {args.pred_dir}   TAP: {args.split}   {n} mau")
    w(f"Loi token toan tap: {100*tot_ed/tot_ref:.2f}%   "
      f"chuoi dung tron ven: {100*(n-n_err_seq)/n:.1f}% ({n-n_err_seq}/{n})")
    w()
    w("== CHUOI SAI HONG O DAU ==")
    w(f"  chi sai ky hieu        : {only_sym:5d}  ({100*only_sym/max(n_err_seq,1):5.1f}% so chuoi sai)")
    w(f"  chi sai quan he        : {only_rel:5d}  ({100*only_rel/max(n_err_seq,1):5.1f}%)")
    w(f"  sai ca hai             : {both:5d}  ({100*both/max(n_err_seq,1):5.1f}%)")
    w()
    w("== PHEP SUA THEO LOAI ==")
    tot_ops = sum(kinds.values())
    name = {"sub": "thay the", "ins": "thua token", "del": "thieu token"}
    for (kind, cat), c in sorted(kinds.items(), key=lambda x: -x[1]):
        w(f"  {name[kind]:12s} {cat:8s} {c:6d}  ({100*c/tot_ops:5.1f}%)")
    w()
    w("== 15 NHAM LAN KY HIEU PHO BIEN NHAT  (du doan -> chuan) ==")
    for (a, b), c in conf.most_common(15):
        w(f"  {a:>14s}  ->  {b:<14s} {c:5d}")
    w()
    w("== 15 NHAM LAN QUAN HE PHO BIEN NHAT ==")
    for (a, b), c in rel_conf.most_common(15):
        w(f"  {a:>14s}  ->  {b:<14s} {c:5d}")
    w()
    w(f"== {args.examples} VI DU (chuoi sai it nhat, de nhin ro cho lech) ==")
    per_seq.sort(key=lambda t: (t[0] == 0, t[0]))
    shown = 0
    for ed, k, ops in per_seq:
        if ed == 0 or shown >= args.examples:
            continue
        shown += 1
        marks = []
        for kind, a, b in ops:
            if kind == "keep":
                marks.append(a)
            elif kind == "sub":
                marks.append(f"[{a} -> {b}]")
            elif kind == "ins":
                marks.append(f"[thua {a}]")
            else:
                marks.append(f"[thieu {b}]")
        w(f"\n--- {k}   ({ed} loi / {len(gt[k])} token)")
        w(f"  chuan   : {' '.join(gt[k])}")
        w(f"  du doan : {' '.join(pred[k])}")
        w(f"  doi chieu: {' '.join(marks)}")

    text = "\n".join(lines)
    if args.out:
        path = full(args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").write(text + "\n")
        print(f"-> {path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
