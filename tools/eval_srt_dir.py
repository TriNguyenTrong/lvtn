"""Score a trained checkpoint against a folder of predicted SRT sequences.

The two-stage variant has one moving part -- the trajectory-to-SRT recogniser --
and swapping it at test time is how we compare recognisers end to end without
retraining the thesis model each time. Every `online/srt_pred*` folder holds one
recogniser's output, one file per split, `basename<TAB>tokens`.

    python tools/eval_srt_dir.py --srt-dir online/srt_pred_thay_v4 \
        --tag srtoof_thayv4 --years 2019

Caveat worth repeating in the write-up: the checkpoint was trained on one
recogniser's noise. Feeding it another recogniser's output measures that
recogniser under a mismatched decoder, which understates a recogniser whose
error pattern differs from the one the decoder learned to expect. Only a
retrained model gives it a fair hearing.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_CKPT = os.path.join(
    "lightning_logs", "abl_dual_shared_srtoof", "lightning_logs", "version_0",
    "checkpoints", "epoch=43-step=50116-val_loss=0.4313.ckpt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="defaults to the model trained on out-of-fold predicted SRT")
    p.add_argument("--srt-dir", required=True)
    p.add_argument("--tag", required=True, help="results/traj_abl_<tag>_<year>_results.txt")
    p.add_argument("--years", default="2014,2016,2019")
    args = p.parse_args()

    from test_all import test_on_dataset

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    srt_dir = args.srt_dir if os.path.isabs(args.srt_dir) else os.path.join(ROOT, args.srt_dir)
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")

    for year in args.years.split(","):
        year = year.strip()
        pred = os.path.join(srt_dir, f"{year}.txt")
        if not os.path.exists(pred):
            print(f"{year}: no predictions in {srt_dir}, skipped", flush=True)
            continue
        out = os.path.join(ROOT, "results", f"traj_abl_{args.tag}_{year}_results.txt")
        print(f"== {year}: {pred} -> {out}", flush=True)
        test_on_dataset(test_year=year, ckpt_path=ckpt, output_file=out,
                        online_input="srt", srt_dir=srt_dir)


if __name__ == "__main__":
    main()
