"""Trajectory -> SRT recogniser: BiLSTM + CTC, after the supervisor's notebook.

Source: https://www.kaggle.com/code/ntcuong2103/math-online-inference/notebook
(CUONG NGUYEN, Apache-2.0). Architecture, feature extraction and loss follow
that notebook; the data plumbing is ours so the model trains on exactly the
8,834-expression split the thesis uses.

Why this matters: its output vocabulary is the same 108 tokens (101 symbols +
7 relations) as vocab/crohme_seq_vocab.txt, which is what the thesis' online
encoder consumed back when it was fed ground-truth SRT. A trained recogniser
therefore turns the oracle variant into a real system without touching the
thesis architecture at all -- predicted SRT simply replaces ground-truth SRT.

    python tools/train_srt_ctc.py --epochs 30
    python tools/train_srt_ctc.py --predict-only      # write predicted SRT
"""
import argparse
import os
import re
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = {"ns": "http://www.w3.org/2003/InkML"}


# --------------------------------------------------------------------------
# vocabulary: 101 symbols, then 7 relations, then the CTC blank
# --------------------------------------------------------------------------
class Vocab:
    def __init__(self, path):
        self.index2word = [w.strip() for w in open(path, encoding="utf-8") if w.strip()]
        self.word2index = {w: i for i, w in enumerate(self.index2word)}
        self.blank = len(self.index2word)          # 108
        self.n_classes = self.blank + 1            # 109

    def encode(self, tokens):
        return [self.word2index[t] for t in tokens if t in self.word2index]

    def decode(self, ids):
        return [self.index2word[i] for i in ids if i < self.blank]


# --------------------------------------------------------------------------
# traces and features, following the notebook exactly
# --------------------------------------------------------------------------
def rdp_simplify(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, matching the `rdp` package the notebook installs.

    Version 4 of the supervisor's notebook simplifies every stroke with
    eps=0.3 on the raw device coordinates, before the height-256 rescale, and
    that halves the point count (476.7 -> 273.0 per expression on CROHME 2019).
    A checkpoint trained that way must be fed the same thing: evaluating his
    published weights without this step cost 3.6 points of token error.

    Iterative rather than recursive: single strokes run to a few hundred
    points, and a pathological one would otherwise blow the stack.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        seg = pts[hi] - pts[lo]
        norm = np.hypot(*seg)
        rel = pts[lo + 1:hi] - pts[lo]
        if norm == 0:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(np.cross(seg, rel)) / norm
        i = int(dist.argmax())
        if dist[i] > eps:
            split = lo + 1 + i
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return pts[keep]


def get_traces(raw: bytes, height: int = 256, rdp_eps: float = 0.0):
    root = ET.fromstring(raw)
    traces = []
    for strk in root.findall("ns:trace", namespaces=NS):
        pts = [p.strip().split()[:2] for p in strk.text.strip().split(",")]
        arr = np.array(pts, dtype="float")
        if len(arr):
            traces.append(rdp_simplify(arr, rdp_eps) if rdp_eps > 0 else arr)
    if not traces:
        return None
    allpts = np.concatenate(traces, 0)
    span = (allpts.max(0) - allpts.min(0))[1] + 1e-6
    ratio = height / span
    return [(t * ratio).astype(int).tolist() for t in traces]


def feature_extraction(traces):
    """[n, 4] = (dx/d, dy/d, d, pen_up); notebook's InkmlDataset.feature_extraction."""
    lengths = [len(t) for t in traces]
    n_points = sum(lengths)
    if n_points < 2:
        return None
    pen_up = np.zeros(n_points - 1)
    idx = np.cumsum(lengths) - 1
    pen_up[idx[:-1]] = 1

    combined = np.concatenate(traces)
    deltas = combined[1:] - combined[:-1]
    d = np.sqrt((deltas ** 2).sum(axis=1))
    feat = np.concatenate([deltas, d[:, None], pen_up[:, None]], axis=1)
    feat = feat[np.where(feat[:, 2] != 0)]         # drop zero-length steps
    if not len(feat):
        return None
    feat[:, :2] /= feat[:, 2][:, None]             # normalise direction
    return feat.astype(np.float32)


# Per-feature statistics of the training split, the same ones the trajectory
# encoder uses. Without them the LSTM sees x at std 3.02 next to dx at std 0.107
# and the stroke-shape channels contribute almost nothing: the unnormalised run
# stalled at 77% token error against 13.9% for the notebook's 4-D vector.
TAP_MEAN = (2.830163, 0.510638, 0.011059, 0.001062, 0.022132, 0.002129, 0.0, 0.0)
TAP_STD = (3.018715, 0.245498, 0.107065, 0.069360, 0.152776, 0.101755, 1.0, 1.0)


def load_tap_features(online_dir, split):
    """The 8-D TAP features already built by tools/prep_online.py.

    Richer than the notebook's 4-D vector: it keeps absolute position, both
    first and second differences, and separate pen-down/pen-up bits. Since the
    SRT recogniser is the bottleneck of the two-stage system, feeding it the
    better representation is the cheapest place to buy accuracy.
    """
    npz = np.load(os.path.join(online_dir, f"{split}.npz"), allow_pickle=False)
    data, off, keys = npz["data"], npz["offsets"], npz["keys"]
    mean = np.asarray(TAP_MEAN, dtype=np.float32)
    std = np.asarray(TAP_STD, dtype=np.float32)
    return {str(k): (np.asarray(data[off[i]:off[i + 1]], dtype=np.float32) - mean) / std
            for i, k in enumerate(keys)}


class InkmlSRT(Dataset):
    def __init__(self, zip_path, split, srt: dict, vocab: Vocab, tap: dict = None,
                 keep: set = None, rdp_eps: float = 0.0):
        self.tap = tap
        self.keep = keep
        self.rdp_eps = rdp_eps
        self.zip_path, self.vocab = zip_path, vocab
        with zipfile.ZipFile(zip_path) as z:
            self.names = sorted(
                n for n in z.namelist()
                if n.startswith(split + "/") and n.lower().endswith(".inkml")
                and os.path.splitext(os.path.basename(n))[0] in srt
                and (keep is None
                     or os.path.splitext(os.path.basename(n))[0] in keep)
                and (tap is None
                     or os.path.splitext(os.path.basename(n))[0] in tap)
            )
        self.srt = srt
        self._zf = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self._zf is None:                       # one handle per worker
            self._zf = zipfile.ZipFile(self.zip_path)
        name = self.names[i]
        base = os.path.splitext(os.path.basename(name))[0]
        if self.tap is not None:
            feat = self.tap.get(base)
        else:
            traces = get_traces(self._zf.read(name), rdp_eps=self.rdp_eps)
            feat = feature_extraction(traces) if traces else None
        if feat is None or not len(feat):
            feat = np.zeros((1, 8 if self.tap is not None else 4), dtype=np.float32)
        lab = np.array(self.vocab.encode(self.srt[base].split()), dtype=np.int64)
        if not len(lab):
            lab = np.zeros(1, dtype=np.int64)
        return torch.from_numpy(feat), torch.from_numpy(lab), len(feat), len(lab), base


def collate(batch):
    feats, labs, flens, llens, bases = zip(*batch)
    fmax = max(flens)
    x = torch.zeros(len(batch), fmax, feats[0].shape[1])
    for i, f in enumerate(feats):
        x[i, : len(f)] = f
    y = torch.cat(labs)
    return x, y, torch.tensor(flens), torch.tensor(llens), bases


class BiLSTMCTC(nn.Module):
    """LSTM_TemporalClassification(4, 128, 3, 109) from the notebook."""

    def __init__(self, n_classes, input_size=4, hidden=128, layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)


def greedy_decode(logits, vocab: Vocab):
    ids = logits.argmax(-1)
    ids = torch.unique_consecutive(ids, dim=-1).tolist()
    return [vocab.index2word[i] for i in ids if i != vocab.blank]


def load_srt(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            out[os.path.splitext(os.path.basename(parts[0]))[0]] = parts[1]
    return out


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_kfold(args, vocab, srt, dev):
    """Out-of-fold SRT for the training split.

    A recogniser trained on all 8,834 expressions gets 2.1% token error when it
    predicts those same expressions back, against 13.9% on unseen data. Training
    the thesis model on that near-perfect signal teaches it to trust the SRT
    stream absolutely, and it then follows the stream off a cliff at test time:
    measured 10.2% ExpRate on the samples whose predicted SRT is imperfect,
    where the same model given correct SRT reaches 66.1%.

    Predicting each fold with a model that never saw it puts the training signal
    at the same error rate as the test signal, so the decoder can learn when to
    fall back on the image branch instead.
    """
    import random

    use_tap = args.features == "tap8"
    tap = load_tap_features(args.online_dir, "train") if use_tap else None
    in_dim = 8 if use_tap else 4
    full = InkmlSRT(args.inkml, "train", srt, vocab, tap=tap, rdp_eps=args.rdp)
    bases = [os.path.splitext(os.path.basename(n))[0] for n in full.names]
    rng = random.Random(7)
    rng.shuffle(bases)
    k = args.kfold
    folds = [set(bases[i::k]) for i in range(k)]
    print(f"{len(bases)} train samples -> {k} folds of ~{len(folds[0])}", flush=True)

    def fit_fold(ds_tr, lr, seed):
        torch.manual_seed(seed)
        model = BiLSTMCTC(vocab.n_classes, input_size=in_dim,
                          hidden=args.hidden, layers=args.layers).to(dev)
        crit = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
        for ep in range(args.epochs):
            model.train()
            tot = n = 0
            for x, y, flens, llens, _ in dl:
                logits = model(x.to(dev))
                loss = crit(logits.log_softmax(-1).permute(1, 0, 2),
                            y.to(dev), flens, llens)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item()
                n += 1
            if ep % 10 == 9 or ep == args.epochs - 1:
                print(f"   epoch {ep:3d}  ctc_loss {tot / max(n, 1):.4f}", flush=True)
        return model

    def decode_fold(model, ds_ho):
        model.eval()
        dl_ho = DataLoader(ds_ho, batch_size=1, shuffle=False,
                           collate_fn=collate, num_workers=0)
        preds, err, ref, short = [], 0, 0, 0
        with torch.no_grad():
            for x, y, flens, llens, b in dl_ho:
                pred = greedy_decode(model(x.to(dev))[0].cpu(), vocab)
                gt = srt[b[0]].split()
                err += levenshtein(pred, gt)
                ref += len(gt)
                short += len(pred) < 0.5 * len(gt)
                preds.append((b[0], pred))
        return preds, 100 * err / max(ref, 1), 100 * short / max(len(preds), 1)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "train.txt")
    all_preds, err, ref = [], 0, 0
    for i, held in enumerate(folds):
        keep_train = set(bases) - held
        ds_tr = InkmlSRT(args.inkml, "train", srt, vocab, tap=tap, keep=keep_train,
                         rdp_eps=args.rdp)
        ds_ho = InkmlSRT(args.inkml, "train", srt, vocab, tap=tap, keep=held,
                         rdp_eps=args.rdp)
        print(f"-- fold {i + 1}/{k}: train {len(ds_tr)}, held out {len(ds_ho)}", flush=True)

        # A fold that collapses onto the CTC blank poisons a fifth of the
        # training signal, and the first tap8 run lost two folds that way
        # (fold 1 at 99.98% token error, fold 2 at 65.16%, against 5-15% for
        # the rest): a 4x256 LSTM on the standardised 8-D features sits close
        # to divergence at lr 1e-3. Halve the rate and retry rather than keep
        # a fold whose predictions are blank.
        lr, seed = args.lr, 1000 + i
        for attempt in range(args.kfold_retries + 1):
            model = fit_fold(ds_tr, lr, seed)
            preds, fold_ter, fold_short = decode_fold(model, ds_ho)
            print(f"   fold {i + 1} attempt {attempt + 1}: lr {lr:g}, "
                  f"token error {fold_ter:.2f}%, short predictions {fold_short:.1f}%",
                  flush=True)
            if fold_ter <= args.kfold_max_ter and fold_short <= 10.0:
                break
            if attempt < args.kfold_retries:
                lr, seed = lr / 2, seed + 500
                print(f"   fold {i + 1} rejected, retrying at lr {lr:g}", flush=True)
        else:
            print(f"   !! fold {i + 1} still bad after "
                  f"{args.kfold_retries + 1} attempts; keeping the last one", flush=True)

        all_preds.extend(preds)
        for base, pred in preds:
            gt = srt[base].split()
            err += levenshtein(pred, gt)
            ref += len(gt)
        print(f"   running out-of-fold token error {100 * err / max(ref, 1):.2f}%", flush=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for base, pred in all_preds:
            f_out.write(f"{base}\t{' '.join(pred)}\n")

    print(f"out-of-fold train SRT: token error rate {100 * err / max(ref, 1):.2f}%"
          f"  -> {out_path}")
    # the test splits keep the predictions of the all-data model: those samples
    # were never in its training set, so there is nothing to correct for
    for split in ("2014", "2016", "2019"):
        src = os.path.join(args.test_src, f"{split}.txt")
        dst = os.path.join(args.out, f"{split}.txt")
        if os.path.exists(src):
            open(dst, "w", encoding="utf-8").write(open(src, encoding="utf-8").read())
            print(f"copied {split} predictions from the all-data model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inkml", default=os.path.join(ROOT, "inkml.zip"))
    p.add_argument("--srt", default=os.path.join(ROOT, "crohme_all.txt"))
    p.add_argument("--vocab", default=os.path.join(ROOT, "vocab", "crohme_seq_vocab.txt"))
    p.add_argument("--out", default=os.path.join(ROOT, "online", "srt_pred"))
    p.add_argument("--ckpt", default=os.path.join(ROOT, "online", "srt_ctc.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--lightning-ckpt",
                   help="load the supervisor's PyTorch Lightning checkpoint instead; "
                        "its keys carry a 'model.' prefix that this strips")
    p.add_argument("--tag", default="", help="suffix for the output folder")
    p.add_argument("--features", default="notebook", choices=["notebook", "tap8"],
                   help="'notebook' = the 4-D vector of the supervisor's notebook; "
                        "'tap8' = the 8-D TAP features from tools/prep_online.py")
    p.add_argument("--rdp", type=float, default=0.0,
                   help="stroke simplification tolerance on raw coordinates; "
                        "0.3 reproduces version 4 of the supervisor's notebook, "
                        "0 the earlier versions and our own runs")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--online-dir", default=os.path.join(ROOT, "online"))
    p.add_argument("--test-src", default=os.path.join(ROOT, "online", "srt_pred"),
                   help="folder whose test-split predictions the k-fold output reuses")
    p.add_argument("--kfold", type=int, default=0,
                   help=">0 writes out-of-fold predictions for the training split")
    p.add_argument("--kfold-retries", type=int, default=2,
                   help="times a collapsed fold is retrained at half the learning rate")
    p.add_argument("--kfold-max-ter", type=float, default=40.0,
                   help="token error above which a fold counts as collapsed")
    args = p.parse_args()
    if args.tag:
        args.out = args.out + "_" + args.tag

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(args.vocab)
    srt = load_srt(args.srt)
    print(f"vocab {len(vocab.index2word)} tokens + blank = {vocab.n_classes} classes")
    print(f"SRT annotations: {len(srt)}")

    use_tap = args.features == "tap8"
    in_dim = 8 if use_tap else 4
    model = BiLSTMCTC(vocab.n_classes, input_size=in_dim,
                      hidden=args.hidden, layers=args.layers).to(dev)
    crit = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    print(f"features={args.features} ({in_dim}-D)  hidden={args.hidden}  layers={args.layers}")

    def tap_for(split):
        return load_tap_features(args.online_dir, split) if use_tap else None

    if args.kfold:
        run_kfold(args, vocab, srt, dev)
        return

    if args.lightning_ckpt:
        raw = torch.load(args.lightning_ckpt, map_location="cpu",
                         weights_only=False)["state_dict"]
        sd = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(sd, strict=True), None
        print(f"loaded {len(sd)} tensors from {os.path.basename(args.lightning_ckpt)}")
        args.predict_only = True

    if not args.predict_only:
        train = InkmlSRT(args.inkml, "train", srt, vocab, tap=tap_for("train"),
                         rdp_eps=args.rdp)
        print(f"train samples: {len(train)}")
        dl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            model.train()
            tot = n = 0
            for x, y, flens, llens, _ in dl:
                x = x.to(dev)
                logits = model(x)
                loss = crit(logits.log_softmax(-1).permute(1, 0, 2),
                            y.to(dev), flens, llens)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item()
                n += 1
            print(f"epoch {ep:3d}  ctc_loss {tot / max(n, 1):.4f}", flush=True)
        torch.save({"state_dict": model.state_dict()}, args.ckpt)
        print(f"saved {args.ckpt}")
    elif not args.lightning_ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=dev)["state_dict"])

    # write predicted SRT for every split, and report token error rate
    model.eval()
    for split in ("train", "2014", "2016", "2019"):
        ds = InkmlSRT(args.inkml, split, srt, vocab, tap=tap_for(split),
                      rdp_eps=args.rdp)
        if not len(ds):
            print(f"{split}: no files, skipped")
            continue
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
        err = ref = 0
        path = os.path.join(args.out, f"{split}.txt")
        with torch.no_grad(), open(path, "w", encoding="utf-8") as f:
            for x, y, flens, llens, bases in dl:
                pred = greedy_decode(model(x.to(dev))[0].cpu(), vocab)
                gt = srt[bases[0]].split()
                err += levenshtein(pred, gt)
                ref += len(gt)
                f.write(f"{bases[0]}\t{' '.join(pred)}\n")
        print(f"{split}: {len(ds)} samples, token error rate {100 * err / max(ref, 1):.2f}%"
              f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
