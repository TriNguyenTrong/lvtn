"""Revision queue G1 -> G2 -> G3: train, pick the best checkpoint, evaluate, summarise.

    python tools/run_revision_queue.py                  # G1, G2, G3 in order
    python tools/run_revision_queue.py --batch G2
    python tools/run_revision_queue.py --only shared_s13
    python tools/run_revision_queue.py --eval-only      # evaluate what already trained
    python tools/run_revision_queue.py --smoke          # G0: every switch, 1 epoch

Resumable: a run whose three results/rev_<tag>_<year>_results.txt exist is skipped,
and a finished training (marker file) is not repeated. A failing run is logged and
the queue moves on. Run it under `conda run -n bttr --no-capture-output`.
"""
import argparse
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
YEARS = ("2014", "2016", "2019")
DIARY = os.path.join(ROOT, "NHAT_KY_CHAY_LAI_2026-09.md")
VAL = ["--val-split", "vocab/val_split_rev.txt"]
AUX = ["--aux-stroke-weight", "0.5", "--suffix", "aux"]
LONG = ["--lr-schedule", "plateau", "--max-epochs", "150",
        "--val-exprate", "--monitor", "val_ExpRate"]
SEEDS = (7, 13, 42)

# fusion arguments of the Table 10 configurations (offline-only carries no aux)
FUSION = {
    "offline": ["--fusion", "offline"],
    "shared": ["--fusion", "dual_shared"] + AUX,
    "cascaded": ["--fusion", "cascaded"] + AUX,
    "concat": ["--fusion", "concat"] + AUX,
}

# Plan of 25/9 evening: every full run uses the long schedule (the 50-epoch schedule
# under-trained the offline baseline by ~6 points on 2014, see the diary).
# offline and shared get 3 seeds (seed 7 is G1); cascaded and concat get seed 7 only.
PLAN = {"G1": [], "G2": [], "G3": [], "G6": []}
PLAN["G1"] += [("long_offline", FUSION["offline"] + LONG, 7),
               ("long_shared", FUSION["shared"] + LONG, 7)]
for name in ("offline", "shared"):
    for sd in (13, 42):
        PLAN["G2"].append((f"{name}_s{sd}", FUSION[name] + LONG, sd))
for name in ("cascaded", "concat"):
    PLAN["G2"].append((f"{name}_s7", FUSION[name] + LONG, 7))
PLAN["G3"] += [
    ("abl_sares", FUSION["shared"] + ["--sa-residual"] + LONG, 7),
    ("abl_sepcross", FUSION["shared"] + ["--separate-cross"] + LONG, 7),
    ("abl_cascx", FUSION["cascaded"] + ["--cascaded-residual", "x"] + LONG, 7),
    # cut on 26/9 for lack of time (a run takes ~8 h): abl_trfenc, online, uni
]
# G6 (29/9, user-approved): the three configurations cut from G3 on 26/9, now with the long
# schedule under NEW tags -- lightning_logs/rev_{uni,online,abl_trfenc} are old stubs from the
# G3 cut and must not be touched or reused; these are separate runs, separate checkpoints.
PLAN["G6"] += [
    ("long_uni", FUSION["shared"] + ["--unidirectional"] + LONG, 7),
    ("long_online", ["--fusion", "online"] + AUX + LONG, 7),
    # long_trfenc only runs if there's time before 30/9 22:00 -- decided at launch time, not here
    ("long_trfenc", FUSION["shared"] + ["--traj-encoder", "transformer"] + LONG, 7),
]
# G1 runs count as seed 7 of the offline / shared groups in the summary
GROUP_OF = {"long_offline": "offline_s7", "long_shared": "shared_s7"}

SMOKE = [
    ("offline", FUSION["offline"]),
    ("shared", FUSION["shared"]),
    ("cascaded", FUSION["cascaded"]),
    ("concat", FUSION["concat"]),
    ("online", ["--fusion", "online"] + AUX),
    ("uni", FUSION["shared"] + ["--unidirectional"]),
    ("sares", FUSION["shared"] + ["--sa-residual"]),
    ("sepcross", FUSION["shared"] + ["--separate-cross"]),
    ("sepcross_casc", FUSION["cascaded"] + ["--separate-cross"]),
    ("cascx", FUSION["cascaded"] + ["--cascaded-residual", "x"]),
    ("trfenc", FUSION["shared"] + ["--traj-encoder", "transformer"]),
    ("plateau_exprate", FUSION["shared"] + ["--lr-schedule", "plateau", "--val-exprate",
                                            "--monitor", "val_ExpRate"]),
    ("plateau_loss", FUSION["offline"] + ["--lr-schedule", "plateau"]),
]

diary_cmd = {}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "rev_queue.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def diary(text):
    with open(DIARY, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def ckpt_dir(tag):
    return os.path.join(ROOT, "lightning_logs", f"rev_{tag}")


def result_path(tag, year):
    return os.path.join(ROOT, "results", f"rev_{tag}_{year}_results.txt")


def best_checkpoint(tag):
    """Best by the NEW validation set: highest val_ExpRate, else lowest val_loss."""
    files = glob.glob(os.path.join(ckpt_dir(tag), "lightning_logs", "version_*", "checkpoints", "*.ckpt"))
    cand = []
    for f in files:
        m = re.search(r"epoch=(\d+)-step=\d+-(val_ExpRate|val_loss)=([0-9.]+)\.ckpt$", f)
        if m:
            cand.append((m.group(2), float(m.group(3)), int(m.group(1)), f))
    if not cand:
        return None, None
    if cand[0][0] == "val_ExpRate":
        best = max(cand, key=lambda c: (c[1], c[2]))
    else:
        best = min(cand, key=lambda c: (c[1], -c[2]))
    return best[3], f"{best[0]}={best[1]:.4f} epoch={best[2]}"


def train(tag, extra, seed, smoke=False):
    d = ckpt_dir(tag)
    marker = os.path.join(d, "TRAIN_DONE")
    if os.path.exists(marker):
        log(f"  training already done for {tag}")
        return True
    if os.path.isdir(d):
        moved = f"{d}_partial_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(d, moved)
        log(f"  unfinished run moved aside to {moved}")
    cmd = [sys.executable, os.path.join(ROOT, "custom_train.py")] + VAL + extra + \
          ["--seed", str(seed), "--out-dir", os.path.relpath(d, ROOT)]
    if "--suffix" not in cmd:
        cmd += ["--suffix", "rev"]
    if smoke:
        cmd += ["--max-epochs", "1", "--check-val-every-n-epoch", "1",
                "--limit-train-batches", "10", "--limit-val-batches", "20"]
    t0 = time.time()
    log(f"  train: {' '.join(cmd[1:])}")
    with open(os.path.join(ROOT, "logs", f"rev_{tag}_train.log"), "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    mins = (time.time() - t0) / 60
    if r.returncode != 0:
        log(f"  !! training {tag} failed (code {r.returncode}), see logs/rev_{tag}_train.log")
        diary(f"### {tag}: HUẤN LUYỆN LỖI (code {r.returncode}) sau {mins:.1f} phút\n"
              f"Lệnh: `{' '.join(cmd[1:])}`\nLog: logs/rev_{tag}_train.log")
        return False
    os.makedirs(d, exist_ok=True)
    open(marker, "w").write(f"{mins:.1f} min\n")
    log(f"  trained {tag} in {mins:.1f} min")
    diary_cmd[tag] = (" ".join(cmd[1:]), mins)
    return True


def evaluate(tag):
    from tools.rev_eval import evaluate as ev
    ckpt, why = best_checkpoint(tag)
    if not ckpt:
        log(f"  !! no checkpoint for {tag}")
        return False
    log(f"  best checkpoint {why}: {ckpt}")
    scores = {}
    t0 = time.time()
    for y in YEARS:
        out = result_path(tag, y)
        if os.path.exists(out):
            continue
        try:
            c, n = ev(ckpt, y, out)
            scores[y] = f"{c}/{n} = {100 * c / n:.2f}%"
        except Exception as e:  # noqa: BLE001
            log(f"  !! evaluation {tag} {y} failed: {e!r}")
            diary(f"### {tag}: ĐÁNH GIÁ {y} LỖI: {e!r}")
            return False
    cmd, mins = diary_cmd.get(tag, ("(huấn luyện ở lần chạy trước)", 0.0))
    diary(f"### {tag}\n- Lệnh: `{cmd}`\n- Huấn luyện: {mins:.1f} phút; đánh giá: {(time.time() - t0) / 60:.1f} phút\n"
          f"- Checkpoint ({why}): `{os.path.relpath(ckpt, ROOT)}`\n- ExpRate: "
          + "; ".join(f"{y}: {scores.get(y, 'đã có từ trước')}" for y in YEARS))
    return True


def parse_result(tag):
    out = {}
    for y in YEARS:
        p = result_path(tag, y)
        if not os.path.exists(p):
            return None
        head = open(p, encoding="utf-8").read(400)
        m = re.search(r"Total: (\d+), Correct: (\d+)", head)
        out[y] = (int(m.group(2)), int(m.group(1)))
    return out


def summarise():
    groups = {}
    for p in glob.glob(os.path.join(ROOT, "results", "rev_*_2014_results.txt")):
        tag = os.path.basename(p)[len("rev_"):-len("_2014_results.txt")]
        res = parse_result(tag)
        if res and not tag.startswith(("smoke", "regress")):
            groups.setdefault(re.sub(r"_s\d+$", "", GROUP_OF.get(tag, tag)), []).append((tag, res))
    lines = ["Revision summary (official denominators 986 / 1147 / 1199, micro over 3332)",
             "tag\t2014\t2016\t2019\tmicro"]
    for g in sorted(groups):
        rows = []
        for tag, res in sorted(groups[g]):
            e = {y: 100 * res[y][0] / res[y][1] for y in YEARS}
            micro = 100 * sum(res[y][0] for y in YEARS) / sum(res[y][1] for y in YEARS)
            rows.append((e, micro))
            lines.append(f"{tag}\t{e['2014']:.2f}\t{e['2016']:.2f}\t{e['2019']:.2f}\t{micro:.2f}")
        if len(rows) == 1 and g in ("offline", "shared", "cascaded", "concat"):
            lines.append(f"{g}: 1 seed (no sd)")
        if len(rows) > 1:
            def ms(vals):
                return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"
            lines.append(f"{g} mean±sd over {len(rows)} seeds\t" +
                         "\t".join(ms([r[0][y] for r in rows]) for y in YEARS) +
                         f"\t{ms([r[1] for r in rows])}")
    open(os.path.join(ROOT, "results", "rev_summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log("summary written to results/rev_summary.txt")


def run_batch(name, only, eval_only):
    for tag, extra, seed in PLAN[name]:
        if only and tag != only:
            continue
        log(f"===== {name} / {tag} =====")
        if all(os.path.exists(result_path(tag, y)) for y in YEARS):
            log("  all three result files exist, skipping")
            continue
        if not eval_only and not train(tag, extra, seed):
            continue
        evaluate(tag)
    summarise()


def run_smoke():
    ok = True
    os.environ["REV_EVAL_LIMIT"] = "15"
    from tools.rev_eval import evaluate as ev
    for tag, extra in SMOKE:
        t = f"smoke_{tag}"
        log(f"===== G0 / {t} =====")
        if not train(t, extra, 7, smoke=True):
            ok = False
            continue
        ckpt, why = best_checkpoint(t)
        if not ckpt:
            log("  !! no checkpoint written")
            ok = False
            continue
        try:
            ev(ckpt, "2014", result_path(t, "2014"))  # strict=True load + beam search
            log(f"  ok ({why})")
        except Exception as e:  # noqa: BLE001
            log(f"  !! eval failed: {e!r}")
            ok = False
    log("G0 " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=["G1", "G2", "G3", "G6"])
    ap.add_argument("--only")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(0 if run_smoke() else 1)
    batches = [args.batch] if args.batch else ["G1", "G2", "G3"]
    if args.only:
        batches = [b for b in batches if any(t == args.only for t, _, _ in PLAN[b])]
    for b in batches:
        run_batch(b, args.only, args.eval_only)
    log("QUEUE DONE")


if __name__ == "__main__":
    main()
