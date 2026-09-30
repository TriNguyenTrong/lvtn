"""CROHME 2023 test set: render offline images from InkML (matched to data.zip's style, per source),
generate the online trajectory .npz, and run inference-only evaluation with the long-schedule checkpoints.

Source: crohme2023_test/INKML/test/CROHME2023_test/*.inkml (official Zenodo 8428035 package,
already extracted; not in git). Does not touch data.zip, inkml.zip, online/{train,2014,2016,2019}.npz,
lightning_logs/, or any existing results/ file. Does not modify any bttr/ model code.

Renderer: verified against CROHME 2014.bmp/data.zip on 2019 (see the diary, 28/9 -- a single global
affine transform, scale~0.650 px/unit, margin~13.5px each side, reproduces data.zip's 2019 bitmaps to
within ~0.5px on 40 random samples). 2023's InkML use a DIFFERENT native coordinate unit per collecting
institution ("copyright" annotation) -- decimal, small magnitude, unrelated to 2014-2019's large-integer
device-pixel coordinates -- so a per-source scale factor is fit before applying the same renderer, per
step B of the instructions: factor_src = (2019 data.zip's median expression pixel height) /
(source's median native-unit expression height). Symbol-level (<traceGroup>) height is NOT used because
2019's own InkML (inkml.zip, the TC-11 package) has NO <traceGroup> at all -- there is nothing to match
it against -- so both sides use expression height, which is the documented fallback ("nếu không có
traceGroup: dùng trung vị chiều cao biểu thức"), applied symmetrically here because the anchor side
(2019) is the one lacking traceGroup, not 2023.

    python tools/prep_crohme2023.py --stage check     # step C: stats, distributions, comparison grid
    python tools/prep_crohme2023.py --stage render     # write online/crohme2023_render/*.png (not data.zip, new dir)
    python tools/prep_crohme2023.py --stage traj        # step D: online/2023.npz
    python tools/prep_crohme2023.py --stage eval         # step E: inference with the 6 long-schedule checkpoints
"""
import argparse
import io
import os
import random
import re
import statistics
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

D2023 = os.path.join(ROOT, "crohme2023_test", "INKML", "test", "CROHME2023_test")
RENDER_DIR = os.path.join(ROOT, "online", "crohme2023_render")   # new dir, no existing file touched
MARGIN = 13.5           # px, each side -- from the 2019 verification fit
LINE_WIDTH = 3          # px -- see step C for the thickness check that motivates this value
NAME_MERGE = {"Nantes_Univertsity_2023": "Nantes_Universite_2023"}  # typo variant, same institution


def parse_inkml(path):
    """-> (source, {trace_id: [(x, y), ...]}). Only <trace> elements; no traceGroup dependency."""
    raw = open(path, encoding="utf-8").read()
    m = re.search(r'<annotation type="copyright">([^<]*)</annotation>', raw)
    src = m.group(1) if m else "(none)"
    src = NAME_MERGE.get(src, src)
    traces = {}
    for tm in re.finditer(r'<trace id="([^"]+)"[^>]*>([^<]+)</trace>', raw):
        pts = []
        for p in tm.group(2).strip().split(","):
            xy = p.strip().split()
            if len(xy) >= 2:
                try:
                    pts.append((float(xy[0]), float(xy[1])))
                except ValueError:
                    pass
        if pts:
            traces[tm.group(1)] = pts
    return src, traces


def bbox_hw(traces):
    xs = [p[0] for t in traces.values() for p in t]
    ys = [p[1] for t in traces.values() for p in t]
    return max(ys) - min(ys), max(xs) - min(xs), min(xs), min(ys)


def all_sources():
    """-> {name: [inkml filenames]}"""
    out = {}
    for f in sorted(os.listdir(D2023)):
        if not f.endswith(".inkml"):
            continue
        src, traces = parse_inkml(os.path.join(D2023, f))
        out.setdefault(src, []).append(f)
    return out


SCALE_2019 = 0.650      # px/native-unit, verified against data.zip on 40 random 2019 samples (28/9 diary)


def source_factors():
    """Per source: median CONTENT (no-margin) pixel height of 2019 / median native height of the source.

    The anchor is 2019's median ink-bbox height (from inkml.zip) times the verified scale SCALE_2019 --
    the CONTENT height, not data.zip's full bitmap height (which adds 2*MARGIN and would double-count
    the margin once MARGIN is added again when rendering 2023)."""
    zi = zipfile.ZipFile(os.path.join(ROOT, "inkml.zip"))
    names_2019 = [n for n in zi.namelist() if n.startswith("2019/") and n.endswith(".inkml")]
    native_h_2019 = []
    for n in names_2019:
        raw = zi.read(n).decode()
        ys = []
        for tm in re.finditer(r'<trace[^>]*>([^<]+)</trace>', raw):
            for p in tm.group(1).strip().split(","):
                xy = p.strip().split()
                if len(xy) >= 2:
                    ys.append(float(xy[1]))
        if ys:
            native_h_2019.append(max(ys) - min(ys))
    anchor = SCALE_2019 * statistics.median(native_h_2019)

    by_src = {}
    for f in sorted(os.listdir(D2023)):
        if not f.endswith(".inkml"):
            continue
        src, traces = parse_inkml(os.path.join(D2023, f))
        if not traces:
            continue
        h, w, _, _ = bbox_hw(traces)
        by_src.setdefault(src, []).append((f, h, w))

    factors = {}
    for src, rows in by_src.items():
        med_h = statistics.median(r[1] for r in rows)
        factors[src] = anchor / med_h
    return factors, anchor, by_src


def render(traces, factor):
    """White strokes on black, binary, no anti-aliasing -- matches data.zip's rendering style."""
    h, w, x0, y0 = bbox_hw(traces)
    W = max(1, round(factor * w + 2 * MARGIN))
    H = max(1, round(factor * h + 2 * MARGIN))
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    r = LINE_WIDTH / 2
    for pts in traces.values():
        xy = [((x - x0) * factor + MARGIN, (y - y0) * factor + MARGIN) for x, y in pts]
        if len(xy) == 1:
            x, y = xy[0]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            continue
        draw.line(xy, fill=255, width=LINE_WIDTH, joint="curve")
        for x, y in xy:                              # round the joints, avoids gaps at sharp turns
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return im


def stroke_thickness(im: Image.Image):
    """Median run length of white pixels, scanned both ways -- a cheap thickness proxy (see 2019 check)."""
    a = np.array(im) > 128
    H, W = a.shape
    runs = []
    for y in range(H):
        row = a[y]
        d = np.diff(np.r_[0, row.view(np.int8), 0])
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    for x in range(W):
        col = a[:, x]
        d = np.diff(np.r_[0, col.view(np.int8), 0])
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    return statistics.median(runs) if runs else 0


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def stage_check(args):
    lines = ["CROHME 2023 test -- step C checks (28/9/2026)."]
    src_files = all_sources()
    lines.append(f"Total InkML files: {sum(len(v) for v in src_files.values())}")
    lines.append("Samples and coordinate-unit type per source (source name as in <annotation "
                 "type=\"copyright\">; '(none)' = tag missing):")
    all_decimal = True
    for src, files in sorted(src_files.items()):
        n_dec = n_int = 0
        for f in files:
            _, traces = parse_inkml(os.path.join(D2023, f))
            if not traces:
                continue
            x0 = next(iter(traces.values()))[0][0]
            if float(x0) != int(float(x0)):
                n_dec += 1
            else:
                n_int += 1
        all_decimal = all_decimal and n_int == 0
        lines.append(f"  {src}: n={len(files)}  decimal-first-coord={n_dec}  integer-first-coord={n_int}")
    lines.append(f"All 2300 samples use decimal coordinates: {all_decimal} "
                 "(a few per source have an integer-valued first coordinate, e.g. '15 20.5' -- still "
                 "the same small-magnitude native unit, not device-pixel scale; checked by inspection)")

    factors, anchor, by_src = source_factors()
    lines.append("")
    lines.append(f"Per-source scale factor (anchor = 2019's median expression CONTENT height, no margin, "
                 f"= {anchor:.1f}px = {SCALE_2019} x median ink-bbox height over all 1,199 2019 InkML in "
                 "inkml.zip; fallback to EXPRESSION height on both sides because inkml.zip's own InkML "
                 "has no <traceGroup> to measure symbol height from):")
    for src, fac in sorted(factors.items()):
        rows = by_src[src]
        med_h = statistics.median(r[1] for r in rows)
        med_w = statistics.median(r[2] for r in rows)
        lines.append(f"  {src}: n={len(rows)}  native median h={med_h:.3f} w={med_w:.3f}  "
                     f"-> factor={fac:.4f} px/unit")

    # render everything once (idempotent) so the distribution and grid can be computed
    os.makedirs(RENDER_DIR, exist_ok=True)
    rendered_h, rendered_w = [], []
    thick_2023 = []
    rng = random.Random(0)
    all_files = [f for files in by_src.values() for f, _, _ in files]
    thick_sample = rng.sample(all_files, min(60, len(all_files)))
    for src, rows in by_src.items():
        fac = factors[src]
        for f, h, w in rows:
            out = os.path.join(RENDER_DIR, f[:-6] + ".png")
            if not os.path.exists(out):
                _, traces = parse_inkml(os.path.join(D2023, f))
                im = render(traces, fac)
                im.save(out)
            else:
                im = Image.open(out)
            rendered_h.append(im.size[1])
            rendered_w.append(im.size[0])
            if f in thick_sample:
                thick_2023.append(stroke_thickness(im))

    zd = zipfile.ZipFile(os.path.join(ROOT, "data.zip"))
    names_2019 = [n for n in zd.namelist() if n.startswith("2019/") and n.endswith(".bmp")]
    h_2019 = [Image.open(io.BytesIO(zd.read(n))).size[1] for n in names_2019]
    w_2019 = [Image.open(io.BytesIO(zd.read(n))).size[0] for n in names_2019]
    thick_2019 = []
    for n in rng.sample(names_2019, min(60, len(names_2019))):
        im = Image.open(io.BytesIO(zd.read(n)))
        thick_2019.append(stroke_thickness(im))

    def stats(v):
        return f"median={statistics.median(v):.1f} p10={pct(v, 0.10):.1f} p90={pct(v, 0.90):.1f}"

    lines.append("")
    lines.append("Rendered-image distribution, 2023 (n=%d) vs 2019/data.zip (n=%d):" % (len(rendered_h), len(h_2019)))
    lines.append(f"  height (px): 2023 {stats(rendered_h)} | 2019 {stats(h_2019)}")
    lines.append(f"  width  (px): 2023 {stats(rendered_w)} | 2019 {stats(w_2019)}")
    lines.append(f"  stroke thickness (median run length, px, {len(thick_2023)} vs {len(thick_2019)} sampled "
                 f"images): 2023 {statistics.median(thick_2023):.2f} | 2019 {statistics.median(thick_2019):.2f} "
                 f"(line_width={LINE_WIDTH}px in the renderer)")

    # comparison grid: 24 random 2023 (rendered) next to 24 random 2019 (data.zip)
    sample_2023 = rng.sample(all_files, 24)
    sample_2019 = rng.sample(names_2019, 24)
    cell = 140
    grid = Image.new("L", (cell * 8, cell * 6), 40)
    for i, f in enumerate(sample_2023):
        im = Image.open(os.path.join(RENDER_DIR, f[:-6] + ".png")).convert("L")
        im.thumbnail((cell - 8, cell - 8))
        x, y = (i % 4) * cell, (i // 4) * cell
        grid.paste(im, (x + 4, y + 4))
    for i, n in enumerate(sample_2019):
        im = Image.open(io.BytesIO(zd.read(n))).convert("L")
        im.thumbnail((cell - 8, cell - 8))
        x, y = (4 + i % 4) * cell, (i // 4) * cell
        grid.paste(im, (x + 4, y + 4))
    grid_path = os.path.join(ROOT, "results", "rev_crohme2023_render_check.png")
    grid.convert("RGB").save(grid_path)
    lines.append("")
    lines.append(f"Comparison grid (left 4 columns = 24 random 2023 renders, right 4 columns = 24 random "
                 f"2019 data.zip bitmaps, seed 0): {os.path.relpath(grid_path, ROOT)}")

    out_txt = os.path.join(ROOT, "results", "rev_crohme2023_render_check.txt")
    open(out_txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out_txt)


def stage_traj(args):
    """Step D: online/2023.npz, via tools/prep_online.py's own build_one (unmodified) -- it
    normalises by expression height, so the per-source pixel-scale factor of step B does not
    apply here (trajectories are scale-invariant by construction)."""
    from prep_online import build_one, report

    step = 0.03
    files = sorted(f for f in os.listdir(D2023) if f.endswith(".inkml"))
    feats, keys, skipped = [], [], []
    for f in files:
        raw = open(os.path.join(D2023, f), "rb").read()
        feat = build_one(raw, step)
        base = f[:-6]
        if feat is None:
            skipped.append(base)
            continue
        feats.append(feat)
        keys.append(base)
    lens = np.array([len(x) for x in feats], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    data = np.concatenate(feats, axis=0).astype(np.float16)
    out = os.path.join(ROOT, "online", "2023.npz")
    np.savez_compressed(out, data=data, offsets=offsets, keys=np.array(keys))
    report("2023", lens, data, skipped)
    print(f"wrote {out}")

    # step D check: average points per expression, resampled, 2023 vs 2019
    d2019 = np.load(os.path.join(ROOT, "online", "2019.npz"), allow_pickle=False)
    lens_2019 = np.diff(d2019["offsets"])
    line = (f"Average resampled points per expression: 2023 {lens.mean():.1f} (n={len(lens)}) "
            f"vs 2019 {lens_2019.mean():.1f} (n={len(lens_2019)})")
    print(line)
    with open(os.path.join(ROOT, "results", "rev_crohme2023_render_check.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n" + line + "\n")


CKPTS = {
    "long_shared": "lightning_logs/rev_long_shared/lightning_logs/version_0/checkpoints/"
                   "epoch=105-step=114904-val_ExpRate=0.8054.ckpt",
    "shared_s13": "lightning_logs/rev_shared_s13/lightning_logs/version_0/checkpoints/"
                  "epoch=69-step=75880-val_ExpRate=0.8213.ckpt",
    "shared_s42": "lightning_logs/rev_shared_s42/lightning_logs/version_0/checkpoints/"
                  "epoch=133-step=145256-val_ExpRate=0.8077.ckpt",
    "long_offline": "lightning_logs/rev_long_offline/lightning_logs/version_0/checkpoints/"
                    "epoch=93-step=101896-val_ExpRate=0.8145.ckpt",
    "offline_s13": "lightning_logs/rev_offline_s13/lightning_logs/version_0/checkpoints/"
                   "epoch=103-step=112736-val_ExpRate=0.8167.ckpt",
    "offline_s42": "lightning_logs/rev_offline_s42/lightning_logs/version_0/checkpoints/"
                   "epoch=131-step=143088-val_ExpRate=0.8122.ckpt",
}
TOK = re.compile(r"\\[a-zA-Z]+|\\.|[^\s\\]")


def load_2023_dataset():
    """-> list of (name, gt_tokens, in_vocab: bool), reading truth + vocabulary membership once."""
    vocab = {l.strip() for l in open(os.path.join(ROOT, "vocab", "dictionary.txt"), encoding="utf-8")}
    out = []
    for f in sorted(os.listdir(D2023)):
        if not f.endswith(".inkml"):
            continue
        raw = open(os.path.join(D2023, f), encoding="utf-8").read()
        m = re.search(r'<annotation type="truth">(.*?)</annotation>', raw, re.S)
        if not m:
            continue
        toks = TOK.findall(m.group(1).strip().strip("$").strip())
        out.append((f[:-6], toks, all(t in vocab for t in toks)))
    return out


def stage_eval(args):
    """Step E: inference only, the six long-schedule checkpoints, both ExpRate levels."""
    import torch
    from PIL import Image as PILImage
    from torchvision.transforms import transforms
    from bttr.datamodule.datamodule import load_online
    from bttr.datamodule.vocab import CROHMEVocab
    from bttr.lit_bttr import LitBTTR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_2023_dataset()
    if args.limit:
        samples = samples[:args.limit]
    n_full = len(samples)
    n_invocab = sum(1 for _, _, ok in samples if ok)
    print(f"2023 test: {n_full} samples with a <truth> annotation, {n_invocab} fully in-vocabulary")
    tags = args.only.split(",") if args.only else list(CKPTS)

    traj = load_online(os.path.join(ROOT, "online", "2023.npz"))
    vocab_dec = CROHMEVocab(os.path.join(ROOT, "vocab", "dictionary.txt"))

    summary = ["CROHME 2023, inference only, six long-schedule checkpoints. Level (a): full 2,300 "
               "samples, a sample with any out-of-vocabulary token counts wrong. Level (b): only the "
               f"{n_invocab} samples whose reference is fully in-vocabulary.",
               "tag\tn(a)\tExpRate(a)%\tn(b)\tExpRate(b)%"]
    prefix = "rev_smoke_" if args.limit else "rev_"
    for tag in tags:
        rel_ckpt = CKPTS[tag]
        out_path = os.path.join(ROOT, "results", f"{prefix}crohme2023_{tag}_results.txt")
        if os.path.exists(out_path):
            print(f"{tag}: already done, skipping")
        else:
            model = LitBTTR.load_from_checkpoint(os.path.join(ROOT, rel_ckpt), strict=True)
            model = model.eval().to(device)
            rows = []
            correct_a = correct_b = 0
            with torch.no_grad():
                for i, (name, gt_toks, in_vocab) in enumerate(samples):
                    img = PILImage.open(os.path.join(RENDER_DIR, name + ".png")).convert("L")
                    img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)
                    img_mask = torch.zeros((1,) + img_t.shape[2:], dtype=torch.bool, device=device)
                    tj = traj.get(name)
                    if tj is None or not len(tj):
                        tj = np.zeros((1, 8), dtype=np.float32)
                    tj_t = torch.from_numpy(tj).unsqueeze(0).to(device)
                    tj_mask = torch.zeros((1, tj_t.shape[1]), dtype=torch.bool, device=device)
                    hyps = model.bttr.beam_search(img_t, img_mask, tj_t, tj_mask,
                                                  model.hparams.beam_size, model.hparams.max_len)
                    best = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
                    pred = vocab_dec.indices2label(best.seq)
                    gt = " ".join(gt_toks)
                    # a sample whose reference has an OOV token can never match (the decoder's
                    # vocabulary cannot produce that token), so this already scores OOV as wrong
                    ok = pred == gt
                    correct_a += ok
                    correct_b += ok and in_vocab
                    rows.append(f"{name}\t{'CORRECT' if ok else 'WRONG'}\t{pred}\t{gt}\t"
                                f"{'invocab' if in_vocab else 'OOV'}")
                    if (i + 1) % 200 == 0:
                        print(f"  {tag}: {i + 1}/{n_full}", flush=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(f"CROHME 2023 test, inference only\n")
                fh.write(f"Level (a) full: {correct_a}/{n_full}, ExpRate {100 * correct_a / n_full:.2f}%\n")
                fh.write(f"Level (b) in-vocab only: {correct_b}/{n_invocab}, "
                        f"ExpRate {100 * correct_b / n_invocab:.2f}%\n")
                fh.write("-" * 40 + "\n")
                fh.write("Image\tStatus\tPrediction\tGroundTruth\tVocab\n")
                fh.write("\n".join(rows))
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        head = open(out_path, encoding="utf-8").read(400)
        ma = re.search(r"\(a\) full: (\d+)/(\d+), ExpRate ([\d.]+)%", head)
        mb = re.search(r"\(b\) in-vocab only: (\d+)/(\d+), ExpRate ([\d.]+)%", head)
        summary.append(f"{tag}\t{ma.group(2)}\t{ma.group(3)}\t{mb.group(2)}\t{mb.group(3)}")
        print(summary[-1])
    open(os.path.join(ROOT, "results", f"{prefix}crohme2023_summary.txt"), "w", encoding="utf-8").write(
        "\n".join(summary) + "\n")
    print("\n".join(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "render", "traj", "eval"])
    ap.add_argument("--limit", type=int, default=0, help="eval: trial run, first N samples")
    ap.add_argument("--only", default=None, help="eval: comma list of tags to run")
    args = ap.parse_args()
    if args.stage in ("check", "render"):   # "render" is a subset of "check"'s work, same renders
        stage_check(args)
    elif args.stage == "traj":
        stage_traj(args)
    elif args.stage == "eval":
        stage_eval(args)


if __name__ == "__main__":
    main()
