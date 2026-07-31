"""Chain the remaining trajectory ablations: train, pick the best checkpoint, evaluate.

Each configuration runs in its own subprocess so no GPU memory carries over.
Everything is sequential -- the GPU fits exactly one job at a time.

    python tools/run_ablations.py            # run the whole plan
    python tools/run_ablations.py --only concat
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YEARS = ("2014", "2016", "2019")

# name -> (train args, checkpoint folder, results tag)
# Every configuration carries the per-stroke symbol loss, because it turned out
# to belong in the main system: on the online-only branch it reached val_loss
# 1.44 by epoch 9, against a best of 2.41 over fifty epochs without it.  An
# ablation is only readable if every row shares that setting.
AUX = ["--aux-stroke-weight", "0.5", "--suffix", "aux"]

PLAN = [
    # already training in another process by the time this runs; evaluate only
    ("online+aux", ["--fusion", "online"] + AUX, "abl_online_aux", "online_aux"),
    # the main system
    ("dual_shared+aux", ["--fusion", "dual_shared"] + AUX,
     "abl_dual_shared_aux", "dual_shared_aux"),
    # training-direction axis
    ("dual_shared_uni+aux", ["--fusion", "dual_shared", "--unidirectional"] + AUX,
     "abl_dual_shared_uni_aux", "dual_shared_uni_aux"),
    # fusion-design axis
    ("concat+aux", ["--fusion", "concat"] + AUX, "abl_concat_aux", "concat_aux"),
    ("cascaded+aux", ["--fusion", "cascaded"] + AUX, "abl_cascaded_aux", "cascaded_aux"),
    # completes the "before" row of the encoder comparison; 2014 already exists
    ("online-noaux-eval", None, "abl_online_traj3", "online_gru"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_checkpoint(folder: str):
    pat = os.path.join(ROOT, "lightning_logs", folder, "lightning_logs",
                       "version_*", "checkpoints", "*.ckpt")
    best, best_v = None, float("inf")
    for f in glob.glob(pat):
        m = re.search(r"val_loss=([0-9.]+)\.ckpt$", f)
        if m and float(m.group(1)) < best_v:
            best, best_v = f, float(m.group(1))
    return best, best_v


def evaluate(ckpt: str, tag: str, years=YEARS):
    sys.path.insert(0, ROOT)
    from test_all import test_on_dataset
    for year in years:
        out = os.path.join(ROOT, "results", f"traj_abl_{tag}_{year}_results.txt")
        if os.path.exists(out):
            log(f"    {year}: already done, skipping")
            continue
        log(f"    evaluating {year}")
        test_on_dataset(test_year=year, ckpt_path=ckpt, output_file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run just this configuration")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate whatever checkpoints exist")
    args = ap.parse_args()

    plan = [p for p in PLAN if not args.only or p[0] == args.only]
    for name, train_args, folder, tag in plan:
        log(f"===== {name} =====")
        ckpt, v = best_checkpoint(folder)
        if train_args is None:
            log("  evaluation only, no training for this entry")
        elif ckpt and not args.eval_only:
            log(f"  checkpoint already exists (val_loss {v:.4f}), skipping training")
        elif not args.eval_only:
            log(f"  training: {' '.join(train_args)}")
            r = subprocess.run([sys.executable, os.path.join(ROOT, "custom_train.py")]
                               + train_args, cwd=ROOT)
            if r.returncode != 0:
                log(f"  !! training failed with code {r.returncode}, moving on")
                continue
            ckpt, v = best_checkpoint(folder)
        if not ckpt:
            log("  !! no checkpoint found, cannot evaluate")
            continue
        log(f"  best checkpoint: val_loss {v:.4f}")
        evaluate(ckpt, tag)
    log("ALL DONE")


if __name__ == "__main__":
    main()
