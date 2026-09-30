"""Evaluate one checkpoint on a CROHME year against the OFFICIAL denominator.

Same TSV layout as test_all.py, with two differences that the revision needs:
  * the checkpoint loads with strict=True, so a switch that silently changes the
    parameter set cannot go unnoticed;
  * Total is the caption count of the test set (986 / 1147 / 1199); a sample the
    pipeline cannot decode (505_em_51 is over 200 tokens, UN19_1001_em_0 has no
    trajectory) is written as a WRONG row instead of vanishing from the count.

    python tools/rev_eval.py <ckpt> <year> <out.txt>
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from bttr.datamodule import CROHMEDatamodule  # noqa: E402
from bttr.lit_bttr import LitBTTR  # noqa: E402


def official_captions(year: str) -> dict:
    with zipfile.ZipFile(os.path.join(ROOT, "data.zip")) as z:
        lines = z.open(f"{year}/caption.txt").read().decode().splitlines()
    out = {}
    for l in lines:
        t = l.strip().split()
        if t:
            out[t[0]] = " ".join(t[1:])
    return out


def evaluate(ckpt: str, year: str, out_path: str, online_input="traj", srt_dir=None):
    model = LitBTTR.load_from_checkpoint(ckpt, strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dm = CROHMEDatamodule(test_year=year, online_input=online_input, srt_dir=srt_dir)
    dm.setup(stage="test")
    rows, seen, correct = [], set(), 0
    limit = int(os.environ.get("REV_EVAL_LIMIT", "0"))  # smoke tests only
    with torch.no_grad():
        for k, batch in enumerate(tqdm(dm.test_dataloader(), mininterval=60)):
            if limit and k >= limit:
                break
            batch = batch.to(device)
            hyps = model.bttr.beam_search(batch.imgs, batch.mask, batch.traj, batch.traj_mask,
                                          model.hparams.beam_size, model.hparams.max_len)
            best = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
            pred = model.vocab_dec.indices2label(best.seq)
            gt = model.vocab_dec.indices2label(batch.indices[0])
            ok = pred == gt
            correct += ok
            seen.add(batch.img_bases[0])
            rows.append(f"{batch.img_bases[0]}\t{'CORRECT' if ok else 'WRONG'}\t{pred}\t{gt}")

    caps = official_captions(year)
    for name, gt in caps.items():
        if name not in seen:
            rows.append(f"{name}\tWRONG\t\t{gt}")
    total = len(caps)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Test Year: {year}\n")
        f.write(f"Total: {total}, Correct: {correct}, ExpRate: {100 * correct / total:.2f}%\n")
        f.write("-" * 40 + "\n")
        f.write("Image\tStatus\tPrediction\tGroundTruth\n")
        f.write("\n".join(rows))
    print(f"{year}: {correct}/{total} = {100 * correct / total:.2f}%  ({total - len(seen)} undecodable counted wrong)")
    return correct, total


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
