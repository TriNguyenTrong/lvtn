"""Per-stroke symbol labels for the online branch's auxiliary loss.

Reads <traceGroup> from the TRAINING InkML only and writes, for every training
expression, one symbol class per pen stroke.  These labels are used as an extra
training signal on the online encoder and are never read at test time, so the
evaluation stays free of ground-truth input -- unlike the SRT variant, where the
labels were fed to the model as its input.

Why it is worth the trouble: the online branch currently learns from 8,834
LaTeX sequences and reaches 47.7% token accuracy, while the image branch reaches
84.5% from the same weak signal.  traceGroup turns that into roughly 143,000
labelled symbols with their stroke boundaries already marked.

    python tools/prep_stroke_labels.py
"""
import argparse
import os
import re
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

import numpy as np


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_symbol_strokes(raw: bytes):
    """-> list of (symbol, [stroke indices]) for the leaf traceGroups."""
    root = ET.fromstring(raw)

    order = []          # trace xml:id in document order
    for el in root.iter():
        if strip_ns(el.tag) == "trace":
            tid = None
            for k, v in el.attrib.items():
                if strip_ns(k) == "id":
                    tid = v
            order.append(tid)
    pos = {tid: i for i, tid in enumerate(order)}

    out = []
    for el in root.iter():
        if strip_ns(el.tag) != "traceGroup":
            continue
        views = [c for c in el if strip_ns(c.tag) == "traceView"]
        if not views:                      # the outer wrapper group
            continue
        label = None
        for c in el:
            if strip_ns(c.tag) == "annotation" and c.attrib.get("type") == "truth":
                label = (c.text or "").strip()
        if not label:
            continue
        idx = []
        for v in views:
            ref = v.attrib.get("traceDataRef")
            if ref in pos:
                idx.append(pos[ref])
        if idx:
            out.append((label, idx))
    return out, len(order)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inkml", default="inkml.zip")
    p.add_argument("--vocab", default="vocab/dictionary.txt")
    p.add_argument("--online", default="online")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    words = [w.strip() for w in open(args.vocab, encoding="utf-8") if w.strip()]
    sym2id = {w: i for i, w in enumerate(words)}
    # CROHME annotates the comparison signs as \lt and \gt; the decoder
    # dictionary spells them out. Same symbol, different notation.
    for a, b in (("\\lt", "<"), ("\\gt", ">")):
        if b in sym2id:
            sym2id[a] = sym2id[b]
    print(f"vocab: {len(words)} symbols")

    # stroke counts the features actually have, to catch any drift
    z = np.load(os.path.join(args.online, f"{args.split}.npz"), allow_pickle=False)
    data, off, keys = z["data"], z["offsets"], z["keys"]
    n_strokes_feat = {}
    for i, k in enumerate(keys):
        n_strokes_feat[str(k)] = int(data[off[i]:off[i + 1], 7].sum())

    labels, offsets, out_keys = [], [0], []
    unknown, n_sym, n_lab, mismatch = Counter(), 0, 0, []
    with zipfile.ZipFile(args.inkml) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.startswith(args.split + "/") and n.lower().endswith(".inkml"))
        for name in names:
            base = os.path.splitext(os.path.basename(name))[0]
            if base not in n_strokes_feat:
                continue
            try:
                pairs, n_tr = parse_symbol_strokes(zf.read(name))
            except ET.ParseError:
                continue
            if n_tr != n_strokes_feat[base]:
                mismatch.append((base, n_tr, n_strokes_feat[base]))
                continue
            arr = np.full(n_tr, -100, dtype=np.int16)   # -100 = ignore in CE
            for sym, idx in pairs:
                n_sym += 1
                if sym not in sym2id:
                    unknown[sym] += 1
                    continue
                for j in idx:
                    arr[j] = sym2id[sym]
                    n_lab += 1
            out_keys.append(base)
            labels.append(arr)
            offsets.append(offsets[-1] + len(arr))

    flat = np.concatenate(labels) if labels else np.zeros(0, np.int16)
    out = os.path.join(args.online, f"stroke_labels_{args.split}.npz")
    np.savez_compressed(out, labels=flat, offsets=np.array(offsets, np.int64),
                        keys=np.array(out_keys))
    print(f"wrote {out}")
    print(f"  expressions      : {len(out_keys)}")
    print(f"  strokes          : {len(flat)}")
    print(f"  labelled strokes : {n_lab}  ({100 * n_lab / max(len(flat), 1):.2f}%)")
    print(f"  symbols seen     : {n_sym}")
    print(f"  stroke-count mismatch (skipped): {len(mismatch)} {mismatch[:5]}")
    if unknown:
        print(f"  symbols not in vocab: {sum(unknown.values())} occurrences, "
              f"{len(unknown)} distinct -> {unknown.most_common(15)}")


if __name__ == "__main__":
    main()
