"""Step 3 of the CROHME 2023 check: labels present? tokens outside the 110-word vocabulary?

Reads the InkML `truth` annotation of crohme2023_test/ (extracted from the official
Zenodo package 8428035, INKML/test/CROHME2023_test) and tokenises it like data.zip.
Writes results/rev_crohme2023_token_check.txt.
"""
import collections
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
voc = {l.strip() for l in open(os.path.join(ROOT, "vocab", "dictionary.txt"), encoding="utf-8")}
TOK = re.compile(r"\\[a-zA-Z]+|\\.|[^\s\\]")


def tok(s):
    return TOK.findall(s.strip().strip("$").strip())


unk, n, bad, lens, bad_samples = collections.Counter(), 0, 0, [], 0
files = glob.glob(os.path.join(ROOT, "crohme2023_test", "INKML", "test", "CROHME2023_test", "*.inkml"))
for f in files:
    d = open(f, encoding="utf-8").read()
    m = re.search(r'<annotation type="truth">(.*?)</annotation>', d, re.S)
    if not m:
        bad += 1
        continue
    t = tok(m.group(1))
    n += 1
    lens.append(len(t))
    u = [w for w in t if w not in voc]
    bad_samples += bool(u)
    unk.update(u)
lines = [f"vocab words: {len(voc)}",
         f"inkml files: {len(files)}; with truth: {n}; without: {bad}",
         f"max length {max(lens)}, over 200 tokens: {sum(l > 200 for l in lens)}",
         f"samples with an out-of-vocabulary token: {bad_samples}/{n}",
         f"distinct OOV tokens: {len(unk)} (occurrences {sum(unk.values())})",
         f"most common OOV: {unk.most_common(40)}"]
print("\n".join(lines))
open(os.path.join(ROOT, "results", "rev_crohme2023_token_check.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
