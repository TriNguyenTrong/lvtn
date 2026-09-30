"""Write vocab/val_split_rev.txt: ~5% of the training set, seed 2026, one name per line.

Also checks the split against the three test sets (name and image bytes), and
that the trajectory and stroke-label archives cover every held-out sample.
"""
import hashlib
import os
import random
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "vocab", "val_split_rev.txt")


def names(z, folder):
    return [l.decode().split()[0] for l in z.open(f"{folder}/caption.txt").readlines() if l.strip()]


def digest(z, folder, n):
    return hashlib.md5(z.read(f"{folder}/{n}.bmp")).hexdigest()


def main():
    z = zipfile.ZipFile(os.path.join(ROOT, "data.zip"))
    train = names(z, "train")
    if len(set(train)) != len(train):
        sys.exit("duplicate names in train")
    rng = random.Random(2026)
    k = round(0.05 * len(train))
    held = sorted(rng.sample(train, k))
    if not os.path.exists(OUT):
        open(OUT, "w", encoding="utf-8").write("\n".join(held) + "\n")
    held = [l.strip() for l in open(OUT, encoding="utf-8") if l.strip()]
    print(f"train {len(train)}  held-out {len(held)}")

    lines = []
    bad = 0
    hset = set(held)
    hh = {digest(z, "train", n): n for n in held}
    for y in ("2014", "2016", "2019"):
        tn = names(z, y)
        by_name = hset & set(tn)
        by_img = [n for n in tn if digest(z, y, n) in hh]
        lines.append(f"{y}: test {len(tn)}  name overlap {len(by_name)}  identical-image overlap {len(by_img)}")
        bad += len(by_name) + len(by_img)
    npz = np.load(os.path.join(ROOT, "online", "train.npz"))
    tr = set(map(str, npz["keys"]))
    st = set(map(str, np.load(os.path.join(ROOT, "online", "stroke_labels_train.npz"))["keys"]))
    lines.append(f"held-out present in trajectories: {len(hset & tr)}/{len(held)}; "
                 f"in stroke labels: {len(hset & st)}/{len(held)}")
    lines.append(f"train kept for training with trajectories: {len(set(train) - hset & tr)}")
    lines.append("RESULT: " + ("OK, no overlap with any test set" if bad == 0 else f"OVERLAP {bad}"))
    print("\n".join(lines))
    open(os.path.join(ROOT, "results", "rev_val_split_check.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
