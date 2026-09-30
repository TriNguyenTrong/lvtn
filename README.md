# Multi-modal Recognition of Handwritten Mathematical Expressions

Code, per-expression results and trained checkpoints for the master's thesis *Multi-modal Approach for Recognizing Handwritten Mathematical Expressions* (Nguyễn Trọng Trí, FPT University, 2026).

The model reads a handwritten expression through two inputs, the rendered image and the recorded pen trajectory, and emits LaTeX. Each input has its own encoder, a DenseNet for the image and a convolutional front end with a BiGRU for the trajectory, and the two memories are fused inside a bidirectionally trained Transformer decoder in which every layer attends to both. The code extends the public implementation of [BTTR](https://github.com/Green-Wood/BTTR) and keeps its package layout.

## Results

ExpRate (%) on the official CROHME test sets, online branch driven by the recorded pen trajectory, mean ± standard deviation over three seeds (7, 13, 42), checkpoint selected on a validation split held out from the training set (Table 10 of the thesis):

| Configuration | CROHME 2014 | CROHME 2016 | CROHME 2019 | All (micro-avg) |
|---|---|---|---|---|
| Offline-only (image) | 52.43 ± 1.27 | 50.39 ± 2.09 | 51.24 ± 1.94 | 51.30 ± 1.30 |
| Shared-query dual cross-attention (proposed) | 57.44 ± 0.68 | 52.37 ± 0.70 | 56.41 ± 0.41 | 55.32 ± 0.39 |

On the CROHME 2023 test set, evaluated without retraining, the proposed model reaches 31.68 ± 0.25% against 29.28 ± 0.79% for the image-only model (Table 12). Every rate is computed over the full test set (986 / 1,147 / 1,199 expressions); an expression that cannot be decoded counts as wrong. The per-expression outputs of the runs are in `results/`.

## Environment

All runs were made on one NVIDIA GeForce RTX 4080 (16 GB) under Windows with Python 3.7.12, PyTorch 1.13.1 (CUDA 11.7) and PyTorch Lightning 1.9.5.

```bash
conda create -n bttr python=3.7
conda activate bttr
# PyTorch 1.13.1 / torchvision 0.14.1, CUDA 11.7 build: see https://pytorch.org/get-started/previous-versions/
pip install -r requirements.txt
pip install -e .
```

`scipy` is needed only for `results/rev_holm_existing.py`.

## Data

| File | Content | Where it comes from |
|---|---|---|
| `data.zip` | CROHME images and LaTeX labels (train, 2014, 2016, 2019) | the archive distributed with BTTR, included here |
| `inkml.zip` | CROHME InkML files, not included | extracted from the ICDAR 2019 CROHME + TFD package of the IAPR TC-11 ([dataset page](https://tc11.cvc.uab.es/datasets/ICDAR2019-CROHME-TDF_1), CC BY-NC-SA 3.0); layout below |
| `crohme_all.txt` | ground-truth symbol-relation sequences, used only by the preliminary experiments | included |
| `vocab/val_split_rev.txt` | the 442 training expressions held out for validation | included; `tools/make_val_split.py` recreates it |

`inkml.zip` must contain four folders, taken from the TC-11 package as follows:

| Folder | Source inside the package | Files |
|---|---|---|
| `train/` | `CROHME2019_data/Task1_onlineRec/MainTask_formula/Train.zip` → `Train/INKMLs/Train_2014/` | 8,834 |
| `2014/` | `.../MainTask_formula/valid.zip` → `valid/TestEM2014GT_INKMLs/` | 986 |
| `2016/` | `CROHME2016_data/Task-1-Formula.zip` → `TEST2016_INKML_noGT/` | 1,147 |
| `2019/` | `.../MainTask_formula/Test.zip` → `TestSet2019/` | 1,199 |

Only the `<trace>` elements are read at test time. The stroke labels of the auxiliary loss come from the `<traceGroup>` annotation of the training files only.

## Reproducing the main system

```bash
# 1. preprocessing, once
python tools/prep_online.py           # -> online/{train,2014,2016,2019}.npz
python tools/prep_stroke_labels.py    # -> online/stroke_labels_train.npz
python tools/make_val_split.py        # -> vocab/val_split_rev.txt (442 names)

# 2. training (shared-query dual cross-attention, seed 7)
python custom_train.py --fusion dual_shared --aux-stroke-weight 0.5 --suffix aux \
    --val-split vocab/val_split_rev.txt --lr-schedule plateau --max-epochs 150 \
    --val-exprate --monitor val_ExpRate --seed 7

# 3. evaluation on one official test set
python tools/rev_eval.py <checkpoint.ckpt> 2019 results/shared_2019.txt
```

Other options of `custom_train.py` select the decoder design (`--fusion dual_shared | cascaded | concat | offline | online`), unidirectional training (`--unidirectional`), the trajectory encoder (`--traj-encoder gru | transformer`), the seed and the decoder-layer ablations (`--sa-residual`, `--separate-cross`, `--cascaded-residual x`). `tools/run_revision_queue.py` lists the runs of Tables 10 and 12 and of Section 4.7.3 and trains, selects and evaluates each of them.

The preliminary experiments (ground-truth symbol-relation input, Tables 3–5) were trained under a 50-epoch schedule with an earlier version of the training script, before its options were kept under version control; their per-expression outputs are in `results/`. The 50-epoch training commands of the two-stage variant in Table 9 are listed in `tools/run_ablations.py` and `tools/run_srtpred_ablation.py`, and `tools/eval_srt_dir.py` evaluates one two-stage checkpoint with each recognizer of Table 7.

## Checkpoints

The checkpoints of the runs in Tables 10 and 12 and of the main-protocol runs of Section 4.7.3 (Table 11, upper part, and the Transformer trajectory encoder) are attached to the release [`thesis-v1`](https://github.com/TriNguyenTrong/lvtn/releases/tag/thesis-v1), with a table (`CHECKPOINTS.md`) that maps each file to its row in the thesis, its epoch, its result files and its SHA-256. The release also holds the checkpoint of the preliminary experiment (Tables 3–5); it was saved by an earlier version of the code and needs the loading step described in `CHECKPOINTS.md`. To evaluate one of the main checkpoints:

```bash
sha256sum -c SHA256SUMS
python tools/rev_eval.py long_shared.ckpt 2014 results/check_2014.txt
```

`tools/rev_eval.py` needs `online/2014.npz` (step 1 above) and loads the checkpoint with strict parameter matching.

## Repository layout

```
custom_train.py         training entry point, one run per call
predict_test.py         recognition of one expression
bttr/                   model: encoders, multi-modal decoder, beam search, data module
tools/                  preprocessing, validation split, training queue, evaluation, analyses
results/                per-expression outputs, summaries and significance tests
vocab/                  dictionaries and the validation split
online/srt_*/           predicted symbol-relation sequences of the two-stage variant
```

## License

The code is released under the MIT License (see `LICENSE`); the parts taken from BTTR keep their original MIT license. The CROHME data, including the images and labels in `data.zip`, are subject to their own terms (the ICDAR 2019 CROHME package is distributed under CC BY-NC-SA 3.0), and the released checkpoints were trained on these data.

## Acknowledgements

The code builds on BTTR by Zhao et al. (ICDAR 2021, [paper](https://link.springer.com/chapter/10.1007%2F978-3-030-86331-9_37), [code](https://github.com/Green-Wood/BTTR)).

## Citation

```bibtex
@mastersthesis{nguyen2026multimodal,
  title  = {Multi-modal Approach for Recognizing Handwritten Mathematical Expressions},
  author = {Nguyễn Trọng Trí},
  school = {FPT University},
  year   = {2026}
}
```
