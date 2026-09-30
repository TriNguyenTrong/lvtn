"""G4a -- re-evaluate the five Table 9 checkpoints with the 8-D recognizer's SRT
(`online/srt_pred_tap8n`) instead of the improved 4-D recognizer used for the published table.

Inference only, no training. Offline-only ignores the online branch entirely (fusion="offline"
never reads feature_online in the decoder) and was trained with online_input="traj", not "srt", so
swapping the SRT source cannot change its output -- the existing `traj_abl_offline_lrmax_*` files are
copied unchanged rather than re-run (same checkpoint, same result by construction).

    python tools/rev_tab9_tap8n.py
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from rev_eval import evaluate  # noqa: E402

YEARS = ("2014", "2016", "2019")
SRT_DIR = os.path.join(ROOT, "online", "srt_pred_tap8n")

RUNS = {
    "online": os.path.join(ROOT, "lightning_logs", "abl_online_srtoof", "lightning_logs",
                           "version_0", "checkpoints", "epoch=37-step=43282-val_loss=1.0438.ckpt"),
    "concat": os.path.join(ROOT, "lightning_logs", "abl_concat_srtoof", "lightning_logs",
                           "version_0", "checkpoints", "epoch=41-step=47838-val_loss=0.4425.ckpt"),
    "cascaded": os.path.join(ROOT, "lightning_logs", "abl_cascaded_srtoof", "lightning_logs",
                             "version_0", "checkpoints", "epoch=43-step=50116-val_loss=0.3929.ckpt"),
    "shared": os.path.join(ROOT, "lightning_logs", "abl_dual_shared_srtoof", "lightning_logs",
                           "version_0", "checkpoints", "epoch=43-step=50116-val_loss=0.4313.ckpt"),
}


def main():
    for year in YEARS:
        src = os.path.join(ROOT, "results", f"traj_abl_offline_lrmax_{year}_results.txt")
        dst = os.path.join(ROOT, "results", f"rev_tab9_tap8n_offline_{year}_results.txt")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print(f"offline {year}: copied unchanged from {os.path.basename(src)} "
                  f"(fusion=offline never reads the online branch)")

    for tag, ckpt in RUNS.items():
        for year in YEARS:
            out = os.path.join(ROOT, "results", f"rev_tab9_tap8n_{tag}_{year}_results.txt")
            if os.path.exists(out):
                print(f"{tag} {year}: already done, skipping")
                continue
            c, n = evaluate(ckpt, year, out, online_input="srt", srt_dir=SRT_DIR)
            print(f"{tag} {year}: {c}/{n} = {100 * c / n:.2f}%")

    lines = ["Table 9 grid re-evaluated with the 8-D recognizer's SRT (online/srt_pred_tap8n) "
             "instead of the improved 4-D recognizer. Same checkpoints as Table 9 (inference only, "
             "no training). Denominators 986/1147/1199.",
             "config\t2014\t2016\t2019\tmicro"]
    for tag in ("offline", "online", "concat", "cascaded", "shared"):
        tot = tots = 0
        cells = []
        for year in YEARS:
            p = os.path.join(ROOT, "results", f"rev_tab9_tap8n_{tag}_{year}_results.txt")
            head = open(p, encoding="utf-8").read(200)
            import re
            m = re.search(r"Total: (\d+), Correct: (\d+)", head)
            n, c = int(m.group(1)), int(m.group(2))
            cells.append(f"{100 * c / n:.2f}")
            tot += c
            tots += n
        lines.append(f"{tag}\t" + "\t".join(cells) + f"\t{100 * tot / tots:.2f}")
    summary = os.path.join(ROOT, "results", "rev_tab9_tap8n_summary.txt")
    open(summary, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", summary)


if __name__ == "__main__":
    main()
