"""G4b -- inference time per expression (review comment 16).

Batch 1, CROHME 2019, three inference modes on the SAME checkpoint:
  a  bidirectional beam 10 (L2R beam + R2L rescoring + R2L beam + L2R rescoring, as in the thesis)
  b  unidirectional beam 10 (L2R only; decoder.bidirectional = False at inference)
  c  greedy (beam 1, unidirectional)
Models: offline-only, main system (shared-query + aux), two-stage (4-D CTC recogniser, live).
Devices: GPU (all 1,199 expressions) and CPU (200 fixed expressions, seed 0; default threads and 4).

The model classes are not touched. The encoder/decoder split replicates BTTR.beam_search
(encoders, optional concat, decoder.beam_search); the offline-only model skips the online
encoder, whose output its decoder never reads. The pre-processing of the pen trajectory from
InkML (tools/prep_online.build_one, or RDP + 4-D features for the two-stage system) is timed on
the raw InkML bytes; its output is discarded and the model reads the stored features, so the
accuracy check is not disturbed by float16 storage. Do NOT run this next to training: the
timings would be wrong. `--limit N` = smoke test (outputs are prefixed rev_smoke_).

    python tools/rev_latency.py --device cuda --models offline,shared_aux,twostage
    python tools/rev_latency.py --device cpu  --threads 0,4
"""
import argparse
import csv
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

MODES = {"a": (10, True), "b": (10, False), "c": (1, False)}       # beam, bidirectional
GROUPS = [("1-10", 1, 10), ("11-20", 11, 20), ("21-30", 21, 30), (">30", 31, 10 ** 9)]
MODELS = {
    "offline": dict(ckpt="lightning_logs/abl_offline_traj3/lightning_logs/version_0/checkpoints/"
                         "epoch=47-step=54672-val_loss=0.4458.ckpt",
                    ref="results/traj_abl_offline_lrmax_2019_results.txt", kind="offline"),
    "shared_aux": dict(ckpt="lightning_logs/abl_dual_shared_aux/lightning_logs/version_0/checkpoints/"
                            "epoch=39-step=45560-val_loss=0.3622.ckpt",
                       ref="results/traj_abl_dual_shared_aux_2019_results.txt", kind="traj"),
    "twostage": dict(ckpt="lightning_logs/abl_dual_shared_srtoof/lightning_logs/version_0/checkpoints/"
                          "epoch=43-step=50116-val_loss=0.4313.ckpt",
                     ref="results/traj_abl_srtoof_thayv4full_2019_results.txt", kind="twostage"),
    "shared_s7": dict(ckpt=None, ref=None, kind="traj"),    # newest G2 checkpoint, if it exists
    # main-protocol (plateau, up to 150 epochs) checkpoints, as released in release_thesis-v1/
    "offline_long": dict(ckpt="release_thesis-v1/long_offline.ckpt",
                         ref="results/rev_long_offline_2019_results.txt", kind="offline"),
    "shared_long": dict(ckpt="release_thesis-v1/long_shared.ckpt",
                        ref="results/rev_long_shared_2019_results.txt", kind="traj"),
}
FIELDS = ["model", "device", "threads", "mode", "name", "n_tokens", "t_pre_ms", "t_rec_ms",
          "t_enc_ms", "t_dec_ms", "t_total_ms", "correct", "pred"]


def cpu_name():
    try:
        out = subprocess.check_output(["wmic", "cpu", "get", "name"], text=True, timeout=20)
        return [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"][0]
    except Exception:  # noqa: BLE001
        return platform.processor()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def read_ref(path):
    d = {}
    if path and os.path.exists(os.path.join(ROOT, path)):
        for line in open(os.path.join(ROOT, path), encoding="utf-8").read().splitlines()[4:]:
            t = line.split("\t")
            if len(t) >= 3:
                d[t[0]] = t[1] == "CORRECT"
    return d


def run_block(name, cfg, dev, threads, modes, sample_idx, limit, out_rows):
    from bttr.datamodule import CROHMEDatamodule
    from bttr.lit_bttr import LitBTTR
    from prep_online import build_one
    from rev_srt_nbest import RECOGNISERS, Recogniser
    from train_srt_ctc import Vocab, feature_extraction, get_traces

    torch.set_num_threads(threads)
    kind = cfg["kind"]
    model = LitBTTR.load_from_checkpoint(os.path.join(ROOT, cfg["ckpt"]), strict=True).eval().to(dev)
    bttr = model.bttr
    if kind == "twostage":
        dm = CROHMEDatamodule(test_year="2019", online_input="srt",
                              srt_dir=os.path.join(ROOT, RECOGNISERS["thay4"]["pred_dir"]))
        rec = Recogniser("thay4", Vocab(os.path.join(ROOT, "vocab", "crohme_seq_vocab.txt")))
        rec.model.to(dev)
    else:
        dm = CROHMEDatamodule(test_year="2019")
        rec = None
    dm.setup(stage="test")
    batches = list(dm.test_dataloader())
    if sample_idx is not None:
        batches = [batches[i] for i in sample_idx]
    if limit:
        batches = batches[:limit]
    zf = zipfile.ZipFile(os.path.join(ROOT, "inkml.zip"))
    raw = {b.img_bases[0]: zf.read(f"2019/{b.img_bases[0]}.inkml") for b in batches
           if kind != "offline"}
    enc_vocab = model.vocab_enc

    def infer(b, beam, bidir, timed):
        b = b.to(dev)
        bttr.decoder.bidirectional = bidir
        t_pre = t_rec = 0.0
        ids = None
        if kind == "traj" and timed:
            sync(dev); t = time.perf_counter()
            build_one(raw[b.img_bases[0]], 0.03)
            t_pre = (time.perf_counter() - t) * 1e3
        if kind == "twostage":
            sync(dev); t = time.perf_counter()
            traces = get_traces(raw[b.img_bases[0]], rdp_eps=RECOGNISERS["thay4"]["rdp"])
            feat = feature_extraction(traces) if traces else None
            if feat is None or not len(feat):
                feat = np.zeros((1, 4), dtype=np.float32)
            t_pre = (time.perf_counter() - t) * 1e3
            sync(dev); t = time.perf_counter()
            with torch.no_grad():
                lp = rec.model(torch.from_numpy(np.asarray(feat, dtype=np.float32))[None].to(dev))[0]
                a = lp.argmax(-1).cpu().numpy()
            keep = np.r_[True, a[1:] != a[:-1]]
            toks = [rec.vocab.index2word[int(i)] for i in a[keep] if i != rec.vocab.blank]
            ids = torch.tensor([enc_vocab.words2indices(toks) or [enc_vocab.PAD_IDX]], device=dev)
            t_rec = (time.perf_counter() - t) * 1e3
        traj = ids if ids is not None else b.traj
        traj_mask = torch.zeros_like(ids, dtype=torch.bool) if ids is not None else b.traj_mask
        with torch.no_grad():
            sync(dev); t0 = time.perf_counter()
            fo, m_img = bttr.encoder_img(b.imgs, b.mask)
            if kind == "offline":
                fon, m_tr = fo, m_img          # memory2 is never read by the offline-only decoder
            else:
                fon, m_tr = bttr.encoder_seq(traj, traj_mask)
            if bttr.fusion == "concat":
                fo = torch.cat((fo, fon), dim=1)
                m_img = torch.cat((m_img, m_tr), dim=1)
            sync(dev); t1 = time.perf_counter()
            hyps = bttr.decoder.beam_search(fo, fon, m_img, m_tr, beam, model.hparams.max_len)
            best = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
            sync(dev); t2 = time.perf_counter()
        pred = model.vocab_dec.indices2label(best.seq)
        gt = model.vocab_dec.indices2label(b.indices[0])
        return pred, gt, t_pre, t_rec, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for mode in modes:
        beam, bidir = MODES[mode]
        for b in batches[:20]:                      # warm-up, not counted
            infer(b, beam, bidir, timed=False)
        for b in batches:
            pred, gt, t_pre, t_rec, t_enc, t_dec = infer(b, beam, bidir, timed=True)
            out_rows.append(dict(model=name, device=dev.type, threads=threads, mode=mode,
                                 name=b.img_bases[0], n_tokens=len(b.indices[0]),
                                 t_pre_ms=f"{t_pre:.3f}", t_rec_ms=f"{t_rec:.3f}",
                                 t_enc_ms=f"{t_enc:.3f}", t_dec_ms=f"{t_dec:.3f}",
                                 t_total_ms=f"{t_pre + t_rec + t_enc + t_dec:.3f}",
                                 correct=int(pred == gt), pred=pred))
        print(f"  {name} {dev.type} thr={threads} mode {mode}: {len(batches)} done", flush=True)
    bttr.decoder.bidirectional = True
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None
    del model, bttr, rec
    import gc
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return peak


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summarise(rows, peaks, prefix, limit):
    from rev_srt_nbest import mcnemar_exact
    cfg_lines = [f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
                 f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}, CPU {cpu_name()}, "
                 f"python {platform.python_version()}, torch default threads {torch.get_num_threads()}",
                 "Batch size 1, CROHME 2019, beam 10 unless greedy. Times in ms per expression: pre = InkML "
                 "-> features (online systems), rec = CTC recogniser (two-stage), enc = image (+ online) encoder, "
                 "dec = beam search incl. cross-scoring. 20 warm-up expressions per block are not counted.",
                 "Mode a = bidirectional beam 10; b = unidirectional beam 10; c = greedy (beam 1, unidirectional).",
                 ""]
    hdr = ("model\tdevice\tthr\tmode\tn\tmean\tmedian\tp95\tpre\trec\tenc\tdec\texpr/s\tExpRate%\t"
           "peakGPU_MB\tmean ms by label length " + " | ".join(g[0] for g in GROUPS))
    lines = [hdr]
    keys = sorted({(r["model"], r["device"], int(r["threads"]), r["mode"]) for r in rows})
    for k in keys:
        rr = [r for r in rows if (r["model"], r["device"], int(r["threads"]), r["mode"]) == k]
        tot = [float(r["t_total_ms"]) for r in rr]
        col = lambda f: statistics.mean(float(r[f]) for r in rr)   # noqa: E731
        by = []
        for _, lo, hi in GROUPS:
            g = [float(r["t_total_ms"]) for r in rr if lo <= int(r["n_tokens"]) <= hi]
            by.append(f"{statistics.mean(g):.0f} (n={len(g)})" if g else "-")
        pk = peaks.get((k[0], k[1]))
        lines.append(f"{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{len(rr)}\t{statistics.mean(tot):.1f}\t"
                     f"{statistics.median(tot):.1f}\t{pct(tot, 0.95):.1f}\t{col('t_pre_ms'):.1f}\t"
                     f"{col('t_rec_ms'):.1f}\t{col('t_enc_ms'):.1f}\t{col('t_dec_ms'):.1f}\t"
                     f"{1000 / statistics.mean(tot):.2f}\t{100 * statistics.mean(int(r['correct']) for r in rr):.2f}\t"
                     f"{'' if pk is None else format(pk, '.0f')}\t" + " | ".join(by))
    lines += ["", "Check of mode (a) on GPU against the stored 2019 result file (per-expression status; "
                  "must be identical, otherwise stop and report):"]
    for m, cfg in MODELS.items():
        ref = read_ref(cfg["ref"])
        rr = [r for r in rows if r["model"] == m and r["device"] == "cuda" and r["mode"] == "a"]
        if not rr or not ref:
            continue
        diff = sum(1 for r in rr if r["name"] in ref and ref[r["name"]] != bool(int(r["correct"])))
        ok = sum(int(r["correct"]) for r in rr)
        lines.append(f"  {m}: {ok}/{len(rr)} = {100 * ok / len(rr):.2f}% now; stored file "
                     f"{sum(ref.values())}/{len(ref)}; expressions whose status differs: {diff}"
                     + ("" if diff == 0 else "   <-- MISMATCH, STOP AND REPORT"))
    lines += ["", "McNemar exact, mode a vs b (same checkpoint, GPU): b = a right & b wrong, c = a wrong & b right"]
    for m in MODELS:
        ra = {r["name"]: int(r["correct"]) for r in rows if r["model"] == m and r["device"] == "cuda" and r["mode"] == "a"}
        rb = {r["name"]: int(r["correct"]) for r in rows if r["model"] == m and r["device"] == "cuda" and r["mode"] == "b"}
        if ra and rb:
            b = sum(1 for n in ra if ra[n] and not rb.get(n, 0))
            c = sum(1 for n in ra if not ra[n] and rb.get(n, 0))
            lines.append(f"  {m}: b={b} c={c} p={mcnemar_exact(b, c):.4g}")
    open(os.path.join(ROOT, "results", prefix + "latency.txt"), "w", encoding="utf-8").write(
        "\n".join(cfg_lines + lines) + "\n")
    print("\n".join(cfg_lines + lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--models", default="offline,shared_aux,twostage")
    ap.add_argument("--modes", default="a,b,c")
    ap.add_argument("--threads", default="0", help="comma list; 0 = torch default (CPU runs: use 0,4)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N expressions")
    ap.add_argument("--prefix", default="rev_", help="output prefix of results/<prefix>latency.{csv,txt}")
    args = ap.parse_args()
    prefix = "rev_smoke_" if args.limit else args.prefix
    csv_path = os.path.join(ROOT, "results", prefix + "latency.csv")
    rows, done = [], set()
    if os.path.exists(csv_path):                    # resume: blocks already measured are kept
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(r["model"], r["device"], int(r["threads"]), r["mode"]) for r in rows}
    peaks = {}
    dev = torch.device(args.device)
    default_threads = torch.get_num_threads()
    sample_idx = None
    if dev.type == "cpu":
        n_all = 1199
        sample_idx = sorted(random.Random(0).sample(range(n_all), 200))
    for name in args.models.split(","):
        cfg = dict(MODELS[name])
        if name == "shared_s7":
            from run_revision_queue import best_checkpoint
            ck, _ = best_checkpoint("long_shared")
            if not ck:
                print("shared_s7: no checkpoint yet, skipped")
                continue
            cfg["ckpt"] = os.path.relpath(ck, ROOT)
        for th in [int(t) for t in args.threads.split(",")]:
            threads = th or default_threads
            modes = [m for m in args.modes.split(",") if (name, dev.type, threads, m) not in done]
            if not modes:
                continue
            print(f"== {name} on {dev.type}, {threads} threads, modes {modes}", flush=True)
            peaks[(name, dev.type)] = run_block(name, cfg, dev, threads, modes, sample_idx, args.limit, rows)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
    summarise(rows, peaks, prefix, args.limit)


if __name__ == "__main__":
    main()
