"""Preprocess CROHME InkML into online trajectory features for the online branch.

Replaces the oracle SRT input (crohme_all.txt) with the raw pen trajectory that a
tablet actually records.  Only <trace> elements are read; <traceGroup>,
<annotation> and <annotationXML> carry the ground truth and are never touched.

Pipeline per expression:
    parse traces -> drop repeated points -> normalise by expression height
    -> resample each stroke at a constant arc-length step -> 8-D point features

The 8-D feature follows TAP (Zhang, Du & Dai, IEEE TMM 2019, eq. 2):
    [x, y, dx, dy, dx', dy', delta(s_i == s_i+1), delta(s_i != s_i+1)]
with dx = x_{i+1} - x_i, dx' = x_{i+2} - x_i, and the last two entries being the
pen-down / pen-up flags.

Output: one .npz per split holding a flat float16 point array plus per-sample
offsets, which keeps 12k variable-length sequences in a single small file.

Usage
-----
    python tools/prep_online.py                       # build online/*.npz
    python tools/prep_online.py --step 0.08           # coarser resampling
    python tools/prep_online.py --preview 8           # sanity-check images only
"""

import argparse
import io
import os
import re
import sys
import zipfile

import numpy as np

# <trace ...> only.  \b does not match between "trace" and "Group", so
# <traceGroup> and <traceFormat> are excluded by construction.
TRACE_RE = re.compile(rb"<trace\b[^>]*>(.*?)</trace>", re.S)

SPLITS = ("train", "2014", "2016", "2019")
FEAT_DIM = 8


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_traces(raw):
    """InkML bytes -> list of strokes, each an [n, 2] float array of (x, y).

    A regex is used rather than an XML parser because a handful of CROHME files
    are not well-formed (see CROHME2019_data/filesWithErrors.txt) yet still hold
    perfectly usable trace data.
    """
    strokes = []
    for body in TRACE_RE.findall(raw):
        pts = []
        for chunk in body.decode("utf-8", "ignore").split(","):
            parts = chunk.split()
            if len(parts) < 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if pts:
            strokes.append(np.asarray(pts, dtype=np.float64))
    return strokes


def drop_repeats(stroke):
    """Remove consecutive duplicate samples (digitisers oversample when idle)."""
    if len(stroke) < 2:
        return stroke
    keep = np.ones(len(stroke), dtype=bool)
    keep[1:] = np.any(np.diff(stroke, axis=0) != 0, axis=1)
    out = stroke[keep]
    return out if len(out) else stroke[:1]


# --------------------------------------------------------------------------- #
# normalisation + resampling
# --------------------------------------------------------------------------- #
def normalise(strokes):
    """Scale the whole expression so its height is 1, keeping the aspect ratio.

    Normalising over the expression (not per stroke) preserves the relative
    position of every symbol -- exactly the structural cue the online branch is
    supposed to contribute.  Degenerate cases (a lone '-' has zero height) fall
    back to the width.
    """
    allpts = np.concatenate(strokes, axis=0)
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    span = hi - lo
    scale = span[1] if span[1] > 1e-9 else (span[0] if span[0] > 1e-9 else 1.0)
    return [(s - lo) / scale for s in strokes]


def resample(stroke, step):
    """Re-sample one stroke at a constant arc-length interval.

    Writing speed and device sampling rate differ wildly across CROHME subsets
    (the 2014 test set is sampled roughly three times as densely as the training
    set), so sampling by distance rather than by time is what keeps the input
    distribution comparable across splits.
    """
    if len(stroke) < 2:
        return stroke
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return stroke[:1]
    n = max(int(np.floor(total / step)) + 1, 2)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, cum, stroke[:, 0]),
                     np.interp(t, cum, stroke[:, 1])], axis=1)


def to_features(strokes):
    """Concatenated strokes -> [N, 8] TAP-style point features."""
    pts = np.concatenate(strokes, axis=0)
    n = len(pts)

    d1 = np.zeros_like(pts)
    d1[:-1] = pts[1:] - pts[:-1]

    d2 = np.zeros_like(pts)
    if n > 2:
        d2[:-2] = pts[2:] - pts[:-2]

    # pen state: 1 if the next sample belongs to the same stroke
    same = np.zeros(n, dtype=np.float64)
    idx = 0
    for s in strokes:
        if len(s):
            same[idx:idx + len(s) - 1] = 1.0
            idx += len(s)
    same[-1] = 0.0  # the very last sample always ends with a pen-up

    return np.concatenate(
        [pts, d1, d2, same[:, None], 1.0 - same[:, None]], axis=1
    ).astype(np.float32)


def build_one(raw, step):
    strokes = parse_traces(raw)
    strokes = [drop_repeats(s) for s in strokes]
    strokes = [s for s in strokes if len(s)]
    if not strokes:
        return None
    strokes = normalise(strokes)
    strokes = [resample(s, step) for s in strokes]
    return to_features(strokes)


# --------------------------------------------------------------------------- #
# driving
# --------------------------------------------------------------------------- #
def build_split(archive, split, step):
    names = sorted(n for n in archive.namelist()
                   if n.startswith(split + "/") and n.lower().endswith(".inkml"))
    feats, keys, skipped = [], [], []
    for name in names:
        f = build_one(archive.read(name), step)
        base = os.path.splitext(os.path.basename(name))[0]
        if f is None:
            skipped.append(base)
            continue
        feats.append(f)
        keys.append(base)
    lens = np.array([len(f) for f in feats], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    data = np.concatenate(feats, axis=0).astype(np.float16)
    return data, offsets, np.array(keys), lens, skipped


def load_online(path):
    """Read one .npz back into {sample name: [N, 8] float32 array}.

    Imported by the datamodule; keep the signature stable.
    """
    z = np.load(path, allow_pickle=False)
    data, offsets, keys = z["data"], z["offsets"], z["keys"]
    return {str(k): np.asarray(data[offsets[i]:offsets[i + 1]], dtype=np.float32)
            for i, k in enumerate(keys)}


def report(split, lens, data, skipped):
    q = lambda p: int(np.percentile(lens, p))
    print(f"  {split:<6} n={len(lens):5d}  len mean={lens.mean():6.1f} "
          f"p50={q(50):5d} p95={q(95):5d} p99={q(99):5d} max={lens.max():5d}  "
          f"{data.nbytes / 1e6:5.1f} MB", flush=True)
    if skipped:
        print(f"         !! {len(skipped)} file khong co trace: {skipped[:5]}")


def cmd_build(args):
    if not os.path.exists(args.inkml):
        sys.exit(f"khong tim thay {args.inkml}")
    os.makedirs(args.out, exist_ok=True)
    print(f"step = {args.step} (don vi = chieu cao bieu thuc)")
    with zipfile.ZipFile(args.inkml) as z:
        for split in SPLITS:
            data, offsets, keys, lens, skipped = build_split(z, split, args.step)
            np.savez_compressed(os.path.join(args.out, f"{split}.npz"),
                                data=data, offsets=offsets, keys=keys)
            report(split, lens, data, skipped)
    print(f"da ghi vao {args.out}/")


def cmd_preview(args):
    """Draw the preprocessed trajectory next to the rendered bitmap.

    This is the one check that catches a wrong filename mapping, a flipped axis
    or a broken normalisation before hours of training are wasted on it.
    """
    from PIL import Image, ImageDraw

    os.makedirs(args.preview_out, exist_ok=True)
    rng = np.random.default_rng(7)
    with zipfile.ZipFile(args.inkml) as z, zipfile.ZipFile(args.images) as zi:
        names = [n for n in z.namelist()
                 if n.startswith("train/") and n.lower().endswith(".inkml")]
        for name in rng.choice(names, size=args.preview, replace=False):
            base = os.path.splitext(os.path.basename(name))[0]
            feat = build_one(z.read(name), args.step)
            if feat is None:
                continue

            H = 200
            xs, ys = feat[:, 0], feat[:, 1]
            W = int(max(xs.max(), 0.1) * H) + 20
            canvas = Image.new("L", (W, H + 20), 255)
            d = ImageDraw.Draw(canvas)
            for i in range(len(feat) - 1):
                if feat[i, 6] > 0.5:  # pen down -> connect to the next sample
                    d.line([(xs[i] * H + 10, ys[i] * H + 10),
                            (xs[i + 1] * H + 10, ys[i + 1] * H + 10)], fill=0, width=2)

            try:
                bmp = Image.open(io.BytesIO(zi.read(f"train/{base}.bmp"))).convert("L")
            except KeyError:
                bmp = Image.new("L", (W, H), 255)
            bmp = bmp.resize((int(bmp.width * H / bmp.height), H))

            out = Image.new("L", (max(canvas.width, bmp.width), canvas.height + bmp.height + 8), 255)
            out.paste(canvas, (0, 0))
            out.paste(bmp, (0, canvas.height + 8))
            out.save(os.path.join(args.preview_out, f"{base}.png"))
            print(f"  {base}: {len(feat)} diem, {int(feat[:, 7].sum())} net")
    print(f"da ghi anh doi chieu vao {args.preview_out}/ (tren = quy dao, duoi = anh render)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inkml", default="inkml.zip")
    p.add_argument("--images", default="data.zip")
    p.add_argument("--out", default="online")
    p.add_argument("--step", type=float, default=0.03,
                   help="arc-length resampling step, in units of expression height; "
                        "0.03 gives ~300 points per expression, ~75 after the 4x "
                        "downsampling inside the encoder")
    p.add_argument("--preview", type=int, default=0,
                   help="draw N sanity-check images instead of building the dataset")
    p.add_argument("--preview-out", default="online/preview")
    args = p.parse_args()

    if args.preview:
        cmd_preview(args)
    else:
        cmd_build(args)


if __name__ == "__main__":
    main()
