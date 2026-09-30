"""Ablation cua bo giai ma khi nhanh online nhan SRT DU DOAN tu net but.

Bang nay lap lai dung luoi cua Bang 4 (offline-only / online-only / concat /
cascaded / dung chung truy van), nhung thay dau vao online tu SRT chuan sang
SRT do mo hinh nhan dang sinh ra.

Cong thuc huan luyen giong het lan chay `abl_dual_shared_srtoof` da co:
  - huan luyen tren SRT du doan OUT-OF-FOLD (`online/srt_pred_oof`, TER tren
    tap huan luyen 7,56%). KHONG dung `online/srt_pred_thay_v4_full/train.txt`:
    file do co TER 2,18% tren tap huan luyen so voi 13-14% tren tap kiem thu,
    tuc la du doan trong mau, se lam bo giai ma qua phu thuoc vao chuoi trung
    gian -- dung loi ma Muc 4.9.2 da mo ta va da xu ly.
  - `--online-dropout 0.2` cho cac cau hinh CO nhanh anh, de bo giai ma con
    biet quay ve doc anh khi chuoi trung gian sai.
  - rieng online-only KHONG dat dropout: khong co nhanh anh de quay ve, che
    chuoi online di thi bo giai ma khong con dau vao nao.

Danh gia: swap bo nhan dang cua thay (`online/srt_pred_thay_v4_full`) vao thoi
diem kiem thu, giong cach da lam cho hang dung chung truy van.

    conda run -n bttr --no-capture-output python tools/run_srtpred_ablation.py
    conda run -n bttr --no-capture-output python tools/run_srtpred_ablation.py --only concat
    conda run -n bttr --no-capture-output python tools/run_srtpred_ablation.py --eval-only
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

TRAIN_SRT_DIR = "online/srt_pred_oof"          # out-of-fold, dung de HUAN LUYEN
TEST_SRT_DIR = "online/srt_pred_thay_v4_full"  # bo nhan dang cua thay, dung de DANH GIA

# ten -> (tham so train, thu muc checkpoint, tag ket qua)
PLAN = [
    ("online", ["--fusion", "online", "--online-input", "srt",
                "--srt-dir", TRAIN_SRT_DIR, "--suffix", "srtoof"],
     "abl_online_srtoof", "srtoof_thayv4_online"),
    ("concat", ["--fusion", "concat", "--online-input", "srt",
                "--srt-dir", TRAIN_SRT_DIR, "--online-dropout", "0.2",
                "--suffix", "srtoof"],
     "abl_concat_srtoof", "srtoof_thayv4_concat"),
    ("cascaded", ["--fusion", "cascaded", "--online-input", "srt",
                  "--srt-dir", TRAIN_SRT_DIR, "--online-dropout", "0.2",
                  "--suffix", "srtoof"],
     "abl_cascaded_srtoof", "srtoof_thayv4_cascaded"),
]

# Hai hang con lai cua bang da co san, khong can chay lai:
#   offline-only        -> results/traj_abl_offline_lrmax_*   (khong dung nhanh online)
#   dung chung truy van -> results/traj_abl_srtoof_thayv4full_*


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_checkpoint(folder):
    pat = os.path.join(ROOT, "lightning_logs", folder, "lightning_logs",
                       "version_*", "checkpoints", "*.ckpt")
    best, best_v = None, float("inf")
    for f in glob.glob(pat):
        m = re.search(r"val_loss=([0-9.]+)\.ckpt$", f)
        if m and float(m.group(1)) < best_v:
            best, best_v = f, float(m.group(1))
    return best, best_v


def evaluate(ckpt, tag, srt_dir, years=YEARS):
    sys.path.insert(0, ROOT)
    from test_all import test_on_dataset
    abs_dir = srt_dir if os.path.isabs(srt_dir) else os.path.join(ROOT, srt_dir)
    for year in years:
        out = os.path.join(ROOT, "results", f"traj_abl_{tag}_{year}_results.txt")
        if os.path.exists(out):
            log(f"    {year}: da co, bo qua")
            continue
        log(f"    danh gia {year} voi {srt_dir}")
        test_on_dataset(test_year=year, ckpt_path=ckpt, output_file=out,
                        online_input="srt", srt_dir=abs_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="chi chay mot cau hinh")
    ap.add_argument("--eval-only", action="store_true",
                    help="bo qua huan luyen, chi danh gia checkpoint dang co")
    ap.add_argument("--also-oof", action="store_true",
                    help="danh gia them voi bo nhan dang out-of-fold cua chung ta")
    args = ap.parse_args()

    plan = [p for p in PLAN if not args.only or p[0] == args.only]
    for name, train_args, folder, tag in plan:
        log(f"===== {name} =====")
        ckpt, v = best_checkpoint(folder)
        if ckpt and not args.eval_only:
            log(f"  da co checkpoint (val_loss {v:.4f}), bo qua huan luyen")
        elif not args.eval_only:
            log(f"  huan luyen: {' '.join(train_args)}")
            r = subprocess.run([sys.executable, os.path.join(ROOT, "custom_train.py")]
                               + train_args, cwd=ROOT)
            if r.returncode != 0:
                log(f"  !! huan luyen loi, ma {r.returncode}, chuyen cau hinh khac")
                continue
            ckpt, v = best_checkpoint(folder)
        if not ckpt:
            log("  !! khong tim thay checkpoint, khong danh gia duoc")
            continue
        log(f"  checkpoint tot nhat: val_loss {v:.4f}")
        evaluate(ckpt, tag, TEST_SRT_DIR)
        if args.also_oof:
            evaluate(ckpt, tag.replace("thayv4", "oof"), TRAIN_SRT_DIR)
    log("XONG")


if __name__ == "__main__":
    main()
