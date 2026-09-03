"""Step 1: can we reproduce the supervisor's own published predictions?

Version 4 of his notebook prints its CROHME 2019 predictions in full, so before
any number of ours is reported back to him we check that his checkpoint, run
through our pipeline, returns the same sequences. The pipeline differs from his
in one place -- he simplifies strokes with RDP(0.3) -- so both settings are run
and compared against his output.
"""
import os
import sys
import zipfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from train_srt_ctc import (Vocab, BiLSTMCTC, get_traces, feature_extraction,
                           greedy_decode, load_srt, levenshtein)

CKPT = os.path.join(ROOT, "srt_ckpt", "epoch17-val_wer0.1368.ckpt")
HIS = os.path.join(ROOT, "online", "srt_pred_thay_v4", "2019.txt")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(os.path.join(ROOT, "vocab", "crohme_seq_vocab.txt"))
    srt = load_srt(os.path.join(ROOT, "crohme_all.txt"))

    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    model = BiLSTMCTC(vocab.n_classes, input_size=4, hidden=128, layers=3)
    model.load_state_dict(sd, strict=True)
    model.to(dev).eval()

    his = {}
    for ln in open(HIS, encoding="utf-8"):
        pa = ln.rstrip("\n").split("\t")
        his[pa[0]] = pa[1].split() if len(pa) > 1 else []

    zf = zipfile.ZipFile(os.path.join(ROOT, "inkml.zip"))
    names = sorted(n for n in zf.namelist()
                   if n.startswith("2019/") and n.lower().endswith(".inkml")
                   and os.path.splitext(os.path.basename(n))[0] in srt)
    blobs = [(os.path.splitext(os.path.basename(n))[0], zf.read(n)) for n in names]
    print(f"{len(blobs)} samples, device {dev}", flush=True)

    for eps in (0.0, 0.3):
        ed = ref = exact = agree = agree_ed = agree_ref = 0
        with torch.no_grad():
            for base, blob in blobs:
                traces = get_traces(blob, rdp_eps=eps)
                feat = feature_extraction(traces) if traces else None
                if feat is None or not len(feat):
                    feat = np.zeros((1, 4), dtype=np.float32)
                x = torch.from_numpy(np.asarray(feat, dtype=np.float32))[None].to(dev)
                pred = greedy_decode(model(x)[0].cpu(), vocab)
                gt = srt[base].split()
                d = levenshtein(pred, gt)
                ed += d; ref += len(gt); exact += (d == 0)
                if base in his:
                    agree += (pred == his[base])
                    agree_ed += levenshtein(pred, his[base])
                    agree_ref += len(his[base])
        tag = f"RDP {eps}" if eps else "khong RDP"
        print(f"{tag:10s}  TER {100*ed/ref:6.2f}%  dung tron {100*exact/len(blobs):5.1f}%"
              f"  |  trung khop output cua thay {100*agree/len(blobs):5.1f}%"
              f"  lech token {100*agree_ed/max(agree_ref,1):5.2f}%", flush=True)
    print("(so sanh: notebook cua thay tu bao TER 13.79% tren 2019)")


if __name__ == "__main__":
    main()
