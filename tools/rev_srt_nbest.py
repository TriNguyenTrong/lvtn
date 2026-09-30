"""G4c -- several SRT hypotheses from the CTC recogniser (review comment 15).

Stages (each is resumable and writes only results/rev_*):

  oracle   CPU. CTC prefix beam search (beam 10, no language model) on the two
           recognisers of the thesis -- the supervisor's 4-D model ('thay4', RDP 0.3,
           the one behind Table 9) and our 8-D model ('tap8n') -- on the three test
           sets. exact@K and oracle TER over K = 1, 3, 5, 10.
           -> results/rev_srt_nbest_oracle.txt  (+ n-best cache in online/srt_nbest_<rec>/)
  entropy  CPU. Mean entropy of the posterior over non-blank frames vs. whether the
           greedy SRT is exact; AUC.  -> results/rev_srt_entropy.txt
  rescore  GPU (do NOT run next to training). The two-stage checkpoint scores each of
           the top-K hypotheses with its own bidirectional beam search; the LaTeX with
           the best length-normalised decoder score wins. lambda = 0 is the result;
           lambda in {0.5, 1} adds lambda * CTC log-probability and is sensitivity only.
           -> results/rev_srt_nbest_rescore.txt (+ per-sample tsv)

    python tools/rev_srt_nbest.py --stage oracle  [--limit 20] [--years 2019]
    python tools/rev_srt_nbest.py --stage rescore --limit 20 --tag smoke

`--limit N` keeps the first N samples per set and prefixes outputs with rev_smoke_ so a
trial run cannot be mistaken for a result.
"""
import argparse
import math
import os
import sys
import zipfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from train_srt_ctc import (BiLSTMCTC, Vocab, feature_extraction, get_traces,  # noqa: E402
                           levenshtein, load_srt, load_tap_features)

YEARS = ("2014", "2016", "2019")
OFFICIAL = {"2014": 986, "2016": 1147, "2019": 1199}
KS = (1, 3, 5, 10)
BEAM = 10
NEG = -float("inf")
PRUNE = math.log(1e-4)          # characters below this posterior are not extended
RECOGNISERS = {
    "thay4": dict(ckpt="srt_ckpt/epoch17-val_wer0.1368.ckpt", features="notebook",
                  hidden=128, layers=3, rdp=0.3, pred_dir="online/srt_pred_thay_v4_full"),
    "tap8n": dict(ckpt="online/srt_ctc_tap8n.pt", features="tap8",
                  hidden=256, layers=4, rdp=0.0, pred_dir="online/srt_pred_tap8n"),
}
TWO_STAGE_CKPT = os.path.join(
    "lightning_logs", "abl_dual_shared_srtoof", "lightning_logs", "version_0",
    "checkpoints", "epoch=43-step=50116-val_loss=0.4313.ckpt")


def lae(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def prefix_beam_search(lp: np.ndarray, blank: int, beam: int = BEAM):
    """CTC prefix beam search over log-probs [T, C]; returns [(tokens, log P(prefix))]."""
    beams = {(): (0.0, NEG)}                      # prefix -> (log p ending in blank, in non-blank)
    for t in range(lp.shape[0]):
        row = lp[t]
        cand = [int(c) for c in np.nonzero(row > PRUNE)[0] if c != blank]
        lb = row[blank]
        new = {}

        def add(prefix, pb=NEG, pnb=NEG):
            ob, onb = new.get(prefix, (NEG, NEG))
            new[prefix] = (lae(ob, pb), lae(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            ptot = lae(pb, pnb)
            add(prefix, pb=ptot + lb)
            last = prefix[-1] if prefix else None
            for c in cand:
                p = row[c]
                if c == last:
                    add(prefix, pnb=pnb + p)          # repeat collapses into the same prefix
                    add(prefix + (c,), pnb=pb + p)    # a blank in between starts a new symbol
                else:
                    add(prefix + (c,), pnb=ptot + p)
        beams = dict(sorted(new.items(), key=lambda kv: -lae(*kv[1]))[:beam])
    return sorted(((list(p), lae(*s)) for p, s in beams.items()), key=lambda x: -x[1])


class Recogniser:
    def __init__(self, name, vocab):
        cfg = RECOGNISERS[name]
        self.name, self.cfg, self.vocab = name, cfg, vocab
        in_dim = 8 if cfg["features"] == "tap8" else 4
        self.model = BiLSTMCTC(vocab.n_classes, input_size=in_dim,
                               hidden=cfg["hidden"], layers=cfg["layers"])
        sd = torch.load(os.path.join(ROOT, cfg["ckpt"]), map_location="cpu", weights_only=False)["state_dict"]
        if name == "thay4":
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self._zip = None
        self._tap = {}

    def samples(self, year, srt):
        """Yield (base, feature array) for the samples the recogniser was evaluated on."""
        if self.cfg["features"] == "tap8":
            tap = self._tap.setdefault(year, load_tap_features(os.path.join(ROOT, "online"), year))
            for base in sorted(tap):
                if base in srt:
                    yield base, tap[base]
        else:
            z = zipfile.ZipFile(os.path.join(ROOT, "inkml.zip"))
            names = sorted(n for n in z.namelist() if n.startswith(year + "/")
                           and n.lower().endswith(".inkml")
                           and os.path.splitext(os.path.basename(n))[0] in srt)
            for n in names:
                base = os.path.splitext(os.path.basename(n))[0]
                traces = get_traces(z.read(n), rdp_eps=self.cfg["rdp"])
                feat = feature_extraction(traces) if traces else None
                if feat is None or not len(feat):
                    feat = np.zeros((1, 4), dtype=np.float32)
                yield base, feat

    @torch.no_grad()
    def log_probs(self, feat):
        x = torch.from_numpy(np.asarray(feat, dtype=np.float32))[None]
        return self.model(x)[0].log_softmax(-1).double().numpy()


def greedy(lp, vocab):
    a = lp.argmax(-1)
    keep = np.r_[True, a[1:] != a[:-1]]
    return [vocab.index2word[int(i)] for i in a[keep] if i != vocab.blank]


def out_name(limit, tag, base):
    return ("rev_smoke_" if limit else "rev_") + base


def cache_path(rec, year, limit):
    d = os.path.join(ROOT, "online", ("smoke_" if limit else "") + f"srt_nbest_{rec}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{year}.tsv")


def build_nbest(rec: Recogniser, year, srt, limit):
    """n-best lists for one set; cached as base<TAB>rank<TAB>logp<TAB>tokens (+ greedy, entropy)."""
    path = cache_path(rec.name, year, limit)
    if os.path.exists(path):
        return path
    vocab = rec.vocab
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for i, (base, feat) in enumerate(rec.samples(year, srt)):
            if limit and i >= limit:
                break
            lp = rec.log_probs(feat)
            g = greedy(lp, vocab)
            p = np.exp(lp)
            nb = lp.argmax(-1) != vocab.blank
            ent = float((-(p * lp).sum(-1))[nb].mean()) if nb.any() else 0.0
            f.write(f"{base}\tG\t{ent:.6f}\t{' '.join(g)}\n")
            for r, (toks, s) in enumerate(prefix_beam_search(lp, vocab.blank)):
                f.write(f"{base}\t{r}\t{s:.6f}\t{' '.join(vocab.index2word[c] for c in toks)}\n")
    os.replace(path + ".tmp", path)
    return path


def read_nbest(path):
    """-> {base: dict(greedy=tokens, entropy=float, nbest=[(tokens, logp)])}"""
    out = {}
    for line in open(path, encoding="utf-8"):
        base, r, v, toks = (line.rstrip("\n").split("\t") + [""])[:4]
        d = out.setdefault(base, dict(greedy=[], entropy=0.0, nbest=[]))
        if r == "G":
            d["greedy"], d["entropy"] = toks.split(), float(v)
        else:
            d["nbest"].append((toks.split(), float(v)))
    return out


def auc(scores, positives):
    """P(score of a random positive > score of a random negative), ties count half."""
    s, y = np.asarray(scores, float), np.asarray(positives, bool)
    if y.all() or not y.any():
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stage_oracle_entropy(args, srt):
    vocab = Vocab(os.path.join(ROOT, "vocab", "crohme_seq_vocab.txt"))
    lines_o = ["CTC n-best ceiling (beam 10, no language model; posterior pruned below 1e-4). "
               "exact@K = share of expressions with at least one hypothesis equal to the reference "
               "SRT; oracle TER = token error of the best hypothesis among the top K.",
               "Denominators for exact@K are the official 986 / 1147 / 1199 (an expression the "
               "recogniser has no file for counts as a miss); TER is over the expressions evaluated.",
               "recogniser\tset\tn_eval\tgreedy exact\tgreedy TER\t" +
               "\t".join(f"exact@{k}" for k in KS) + "\t" + "\t".join(f"oracleTER@{k}" for k in KS)]
    lines_e = ["Mean posterior entropy over non-blank frames (nats) vs. greedy-SRT exactness. "
               "AUC = P(wrong sample has higher entropy than a right one); 0.5 = no information.",
               "recogniser\tset\tn\tSRT exact %\tmean H (right)\tmean H (wrong)\tAUC"]
    tot = {}
    for name in args.recognisers:
        rec = Recogniser(name, vocab)
        for year in args.years:
            nb = read_nbest(build_nbest(rec, year, srt, args.limit))
            n = len(nb)
            ex = {k: 0 for k in KS}
            ted = {k: 0 for k in KS}
            ref_tokens = g_ex = g_ed = 0
            ents, wrong = [], []
            for base, d in nb.items():
                gt = srt[base].split()
                ref_tokens += len(gt)
                gd = levenshtein(d["greedy"], gt)
                g_ex += gd == 0
                g_ed += gd
                ents.append(d["entropy"])
                wrong.append(gd > 0)
                dists = [levenshtein(h, gt) for h, _ in d["nbest"]]
                for k in KS:
                    m = min(dists[:k]) if dists else len(gt)
                    ex[k] += m == 0
                    ted[k] += m
            den = OFFICIAL[year] if not args.limit else n
            lines_o.append(
                f"{name}\t{year}\t{n}\t{100 * g_ex / den:.2f}\t{100 * g_ed / max(ref_tokens, 1):.2f}\t" +
                "\t".join(f"{100 * ex[k] / den:.2f}" for k in KS) + "\t" +
                "\t".join(f"{100 * ted[k] / max(ref_tokens, 1):.2f}" for k in KS))
            e = np.array(ents)
            w = np.array(wrong)
            lines_e.append(f"{name}\t{year}\t{n}\t{100 * (1 - w.mean()):.2f}\t"
                           f"{e[~w].mean() if (~w).any() else float('nan'):.4f}\t"
                           f"{e[w].mean() if w.any() else float('nan'):.4f}\t{auc(e, w):.4f}")
            t = tot.setdefault(name, dict(den=0, g=0, ex={k: 0 for k in KS}))
            t["den"] += den
            t["g"] += g_ex
            for k in KS:
                t["ex"][k] += ex[k]
            print(lines_o[-1], flush=True)
    if len(args.years) == 3:
        for name, t in tot.items():
            lines_o.append(f"{name}\tmicro\t-\t{100 * t['g'] / t['den']:.2f}\t-\t" +
                           "\t".join(f"{100 * t['ex'][k] / t['den']:.2f}" for k in KS) + "\t-\t-\t-\t-")
    for text, fn in ((lines_o, "srt_nbest_oracle"), (lines_e, "srt_entropy")):
        p = os.path.join(ROOT, "results", out_name(args.limit, args.tag, fn + ".txt"))
        open(p, "w", encoding="utf-8").write("\n".join(text) + "\n")
        print("wrote", p)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    term, tot = 1, 1                       # math.comb needs Python 3.8; the env is 3.7
    for i in range(1, k + 1):
        term = term * (n - i + 1) // i
        tot += term
    p = tot / 2 ** n
    return min(1.0, 2 * p)


def baseline_correct(year):
    """{name: bool} from the existing Table-9 result file, missing samples counted wrong."""
    from tools.rev_eval import official_captions
    d = {n: False for n in official_captions(year)}
    p = os.path.join(ROOT, "results", f"traj_abl_srtoof_thayv4full_{year}_results.txt")
    for line in open(p, encoding="utf-8").read().splitlines()[4:]:
        t = line.split("\t")
        if len(t) >= 3:
            d[t[0]] = t[1] == "CORRECT"
    return d


def stage_rescore(args, srt):
    from bttr.datamodule import CROHMEDatamodule
    from bttr.lit_bttr import LitBTTR
    from tools.rev_eval import official_captions
    vocab = Vocab(os.path.join(ROOT, "vocab", "crohme_seq_vocab.txt"))
    rec = Recogniser("thay4", vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitBTTR.load_from_checkpoint(os.path.join(ROOT, args.ckpt or TWO_STAGE_CKPT), strict=True)
    model = model.eval().to(device)
    K = args.k
    lambdas = (0.0, 0.5, 1.0)
    rows, summary = [], {}
    sample_lines = ["set\tname\tk\tctc_logp\tdec_score\tcorrect\tlatex"]
    for year in args.years:
        nb = read_nbest(build_nbest(rec, year, srt, args.limit))
        dm = CROHMEDatamodule(test_year=year, online_input="srt",
                              srt_dir=os.path.join(ROOT, RECOGNISERS["thay4"]["pred_dir"]))
        dm.setup(stage="test")
        picked = {lam: {} for lam in lambdas}
        cand_ok = {}
        first_ok = {}
        n_done = 0
        with torch.no_grad():
            for batch in dm.test_dataloader():
                name = batch.img_bases[0]
                if name not in nb or not nb[name]["nbest"]:
                    continue
                if args.limit and n_done >= args.limit:
                    break
                n_done += 1
                batch = batch.to(device)
                gt = model.vocab_dec.indices2label(batch.indices[0])
                cands = []
                for r, (toks, lpc) in enumerate(nb[name]["nbest"][:K]):
                    if not toks:
                        continue
                    ids = torch.tensor([vocab_enc_ids(model, toks)], device=device)
                    tmask = torch.zeros_like(ids, dtype=torch.bool)
                    hyps = model.bttr.beam_search(batch.imgs, batch.mask, ids, tmask,
                                                  model.hparams.beam_size, model.hparams.max_len)
                    best = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
                    latex = model.vocab_dec.indices2label(best.seq)
                    score = float(best.score / (len(best) ** model.hparams.alpha))
                    cands.append((r, lpc, score, latex))
                    sample_lines.append(f"{year}\t{name}\t{r}\t{lpc:.4f}\t{score:.4f}\t{int(latex == gt)}\t{latex}")
                if not cands:
                    continue
                first_ok[name] = cands[0][3] == gt
                cand_ok[name] = any(c[3] == gt for c in cands)
                for lam in lambdas:
                    win = max(cands, key=lambda c: c[2] + lam * c[1])
                    picked[lam][name] = win[3] == gt
        caps = official_captions(year)
        base = baseline_correct(year)
        if args.limit:                       # trial run: compare only what was decoded
            caps = {n: "" for n in cand_ok}
            base = {n: base[n] for n in cand_ok}
        summary[year] = dict(n=OFFICIAL[year] if not args.limit else n_done, base=base, picked=picked,
                             first=first_ok, oracle=cand_ok, caps=caps)
        print(year, "done", n_done, flush=True)

    lines = [f"Re-scoring the top-{K} CTC hypotheses of the 4-D recogniser with the two-stage checkpoint "
             f"{args.ckpt or TWO_STAGE_CKPT} (bidirectional beam 10 per hypothesis, no training).",
             "Selection: highest length-normalised decoder score (lambda = 0, the result). lambda = 0.5 and 1 add "
             "lambda x CTC log-probability of the hypothesis: sensitivity only, not tuned on any data.",
             "'top-1 only' = hypothesis rank 0 alone (beam-search top-1, not the greedy SRT of Table 9).",
             "'best-of-K oracle' = at least one of the K decoded LaTeX is exact (upper bound for any selector).",
             "Denominators 986 / 1147 / 1199; an expression with no hypothesis counts wrong. Baseline = Table 9 "
             "shared-query row (results/traj_abl_srtoof_thayv4full_*), same denominators.",
             "row\t" + "\t".join(YEARS) + "\tmicro\tvs baseline (b,c,p exact McNemar over 3 sets)"]
    def line(label, getter):
        tot = tots = 0
        cells = []
        b = c = 0
        for y in YEARS:
            if y not in summary:
                cells.append("-")
                continue
            s = summary[y]
            ok = getter(s)
            corr = sum(1 for v in ok.values() if v)
            cells.append(f"{100 * corr / s['n']:.2f}")
            tot += corr
            tots += s["n"]
            for nme in s["caps"]:
                x, z = ok.get(nme, False), s["base"].get(nme, False)
                b += x and not z
                c += z and not x
        return f"{label}\t" + "\t".join(cells) + f"\t{100 * tot / max(tots, 1):.2f}\t{b}/{c} p={mcnemar_exact(b, c):.4g}"
    lines.append(line("baseline (Table 9 shared-query)", lambda s: s["base"]))
    lines.append(line("top-1 only", lambda s: s["first"]))
    for lam in lambdas:
        tag = "lambda=0 (RESULT)" if lam == 0 else f"lambda={lam} (sensitivity)"
        lines.append(line(f"top-{K}, {tag}", lambda s, lam=lam: s["picked"][lam]))
    lines.append(line(f"best-of-{K} oracle", lambda s: s["oracle"]))
    p = os.path.join(ROOT, "results", out_name(args.limit, args.tag, "srt_nbest_rescore.txt"))
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(p.replace(".txt", "_samples.tsv"), "w", encoding="utf-8").write("\n".join(sample_lines) + "\n")
    print("\n".join(lines))


def vocab_enc_ids(model, toks):
    return model.vocab_enc.words2indices(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["oracle", "entropy", "rescore"], default="oracle")
    ap.add_argument("--years", default="2014,2016,2019")
    ap.add_argument("--recognisers", default="thay4,tap8n")
    ap.add_argument("--limit", type=int, default=0, help="trial run: first N samples per set")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    args.years = args.years.split(",")
    args.recognisers = args.recognisers.split(",")
    torch.set_num_threads(args.threads)
    srt = load_srt(os.path.join(ROOT, "crohme_all.txt"))
    if args.stage == "rescore":
        stage_rescore(args, srt)
    else:
        stage_oracle_entropy(args, srt)


if __name__ == "__main__":
    main()
