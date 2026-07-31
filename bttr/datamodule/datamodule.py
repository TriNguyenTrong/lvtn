import os
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import BoolTensor, FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from .vocab import CROHMEVocab

# vocab = CROHMEVocab()

Data = List[Tuple[str, Image.Image, np.ndarray, List[int]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# Online trajectories are two orders of magnitude longer than the symbol
# sequences this branch used to read, so batches need a length budget of their
# own: a small image can still carry a very long trajectory.
TRAJ_DIM = 8
MAX_TRAJ_SIZE = 25600  # points x batch, i.e. 32 samples of 800 points
# Raising this also means raising max_len of SeqEncoder, which sizes its
# positional-encoding table from it.
MAX_TRAJ_LEN = 3000

# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
    batch_Trajsize: int = MAX_TRAJ_SIZE,
    maxTrajlen: int = MAX_TRAJ_LEN,
):
    fname_batch = []
    feature_batch = []
    traj_batch = []
    label_batch = []
    stroke_batch = []
    feature_total = []
    traj_total = []
    label_total = []
    stroke_total = []
    fname_total = []
    biggest_image_size = 0
    biggest_traj_len = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, traj, lab, stroke in data:
        size = fea.size[0] * fea.size[1]
        fea = transforms.ToTensor()(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        if len(traj) > biggest_traj_len:
            biggest_traj_len = len(traj)
        batch_image_size = biggest_image_size * (i + 1)
        batch_traj_size = biggest_traj_len * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[1]} x {fea.shape[2]} =  bigger than {maxImagesize}, ignore"
            )
        elif len(traj) > maxTrajlen:
            print(f"trajectory: {fname} length {len(traj)} bigger than {maxTrajlen}, ignore")
        else:
            if (
                batch_image_size > batch_Imagesize
                or batch_traj_size > batch_Trajsize
                or i == batch_size
            ):  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                traj_total.append(traj_batch)
                label_total.append(label_batch)
                stroke_total.append(stroke_batch)
                i = 0
                biggest_image_size = size
                biggest_traj_len = len(traj)
                fname_batch = []
                feature_batch = []
                traj_batch = []
                label_batch = []
                stroke_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                traj_batch.append(traj)
                label_batch.append(lab)
                stroke_batch.append(stroke)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                traj_batch.append(traj)
                label_batch.append(lab)
                stroke_batch.append(stroke)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    traj_total.append(traj_batch)
    label_total.append(label_batch)
    stroke_total.append(stroke_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, traj_total, label_total, stroke_total))


def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"{dir_name}/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


def load_online(path: str) -> dict:
    """Read one .npz produced by tools/prep_online.py.

    The file stores every trajectory in a single flat array plus per-sample
    offsets; see tools/prep_online.py for how it is written.
    """
    npz = np.load(path, allow_pickle=False)
    data, offsets, keys = npz["data"], npz["offsets"], npz["keys"]
    return {
        str(k): np.asarray(data[offsets[i] : offsets[i + 1]], dtype=np.float32)
        for i, k in enumerate(keys)
    }


def load_srt(path: str, vocab_enc) -> dict:
    """Read `name <TAB> token token ...` and encode with the encoder vocabulary.

    The file is written by tools/train_srt_ctc.py, so the sequence is PREDICTED
    from the pen trajectory, not read from the InkML annotation. Same format
    either way, which is the point: the model cannot tell the difference, so the
    two variants differ in exactly one thing, the quality of the sequence.
    """
    out = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2 or not parts[1].strip():
            continue
        base = os.path.splitext(os.path.basename(parts[0]))[0]
        out[base] = np.asarray(vocab_enc.words2indices(parts[1].split()), dtype=np.int64)
    return out


def load_stroke_labels(path: str) -> dict:
    """Per-stroke symbol ids written by tools/prep_stroke_labels.py (train only)."""
    npz = np.load(path, allow_pickle=False)
    labels, offsets, keys = npz["labels"], npz["offsets"], npz["keys"]
    return {
        str(k): np.asarray(labels[offsets[i]: offsets[i + 1]], dtype=np.int64)
        for i, k in enumerate(keys)
    }


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    traj: FloatTensor  # [b, l1, 8]
    traj_mask: BoolTensor  # [b, l1], True = padding
    indices: List[List[int]]  # [b, l2]
    stroke_labels: Optional[LongTensor] = None  # [b, s], -100 = ignore

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            traj=self.traj.to(device),
            traj_mask=self.traj_mask.to(device),
            indices=self.indices,
            stroke_labels=None if self.stroke_labels is None
            else self.stroke_labels.to(device),
        )


def augment_traj(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random affine distortion of one trajectory, applied to the 8-D features.

    The deltas are linear functions of the coordinates, so a single 2x2 map A
    applies unchanged to (x, y), (dx, dy) and (d'x, d'y) -- no need to rebuild
    the features from raw points, and no change to tools/prep_online.py.  The
    two pen bits are untouched.  Translation moves only the absolute coordinates,
    which is the point: it is absolute position that lets the encoder identify a
    training sample instead of reading its shape.
    """
    theta = rng.uniform(-8.0, 8.0) * np.pi / 180.0
    sx, sy = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    shear = rng.uniform(-0.2, 0.2)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    A = rot @ np.array([[sx, shear], [0.0, sy]], dtype=np.float32)

    out = np.asarray(traj, dtype=np.float32).copy()
    for i in (0, 2, 4):                      # (x,y), (dx,dy), (d'x,d'y)
        out[:, i:i + 2] = out[:, i:i + 2] @ A.T
    out[:, 0] += rng.uniform(-0.3, 0.3)
    out[:, 1] += rng.uniform(-0.1, 0.1)
    return out


def collate_fn(batch, augment: bool = False):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    trajs_x = batch[2]
    seqs_y = batch[3]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # srt mode carries 1-D integer token ids; traj mode carries [n, 8] point features
    is_srt = np.asarray(trajs_x[0]).ndim == 1
    lens_t = [len(t) for t in trajs_x]
    max_len_t = max(lens_t)
    if is_srt:
        t = torch.zeros(n_samples, max_len_t, dtype=torch.long)
        t_mask = torch.ones(n_samples, max_len_t, dtype=torch.bool)
        for idx, s_t in enumerate(trajs_x):
            t[idx, : lens_t[idx]] = torch.from_numpy(np.asarray(s_t, dtype=np.int64))
            t_mask[idx, : lens_t[idx]] = False
    else:
        if augment:
            rng = np.random.default_rng()
            trajs_x = [augment_traj(t, rng) for t in trajs_x]
            lens_t = [len(t) for t in trajs_x]
            max_len_t = max(lens_t)
        t = torch.zeros(n_samples, max_len_t, TRAJ_DIM)
        t_mask = torch.ones(n_samples, max_len_t, dtype=torch.bool)
        for idx, s_t in enumerate(trajs_x):
            t[idx, : lens_t[idx]] = torch.from_numpy(np.asarray(s_t, dtype=np.float32))
            t_mask[idx, : lens_t[idx]] = False

    # per-stroke symbol labels, present for the training split only
    labels = None
    if len(batch) > 4 and batch[4] is not None and any(l is not None for l in batch[4]):
        lens_s = [0 if l is None else len(l) for l in batch[4]]
        labels = torch.full((n_samples, max(lens_s)), -100, dtype=torch.long)
        for idx, l in enumerate(batch[4]):
            if l is not None and len(l):
                labels[idx, : len(l)] = torch.from_numpy(np.asarray(l, dtype=np.int64))

    return Batch(fnames, x, x_mask, t, t_mask, seqs_y, labels)


def build_dataset(archive, traj_dict, folder: str, vocab_dec, batch_size: int,
                  stroke_dict: Optional[dict] = None):
    data = extract_data(archive, folder)
    stroke_dict = stroke_dict or {}
    data = [(fname, img, traj_dict[fname], vocab_dec.words2indices(formula),
             stroke_dict.get(fname))
    for fname, img, formula in data if fname in traj_dict]
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        online_dir: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../online",
        vocab_dec: str = "vocab/dictionary.txt",
        test_year: str = "2014",
        batch_size: int = 8,
        num_workers: int = 5,
        online_input: str = "traj",
        srt_dir: str = None,
        vocab_enc: str = "vocab/crohme_seq_vocab.txt",
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        assert online_input in ("traj", "srt")
        self.zipfile_path = zipfile_path
        self.online_dir = online_dir
        self.online_input = online_input
        # default: SRT predicted from the trajectory by the BiLSTM-CTC recogniser
        self.srt_dir = srt_dir or os.path.join(online_dir, "srt_pred_thay")
        self.vocab_enc = CROHMEVocab(vocab_enc) if online_input == "srt" else None
        self.test_year = test_year
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_dec = CROHMEVocab(vocab_dec)

        print(f"Load data from: {self.zipfile_path}")
        print(f"Load online trajectories from: {self.online_dir}")

    def _traj(self, split: str) -> dict:
        if self.online_input == "srt":
            path = os.path.join(self.srt_dir, f"{split}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} khong ton tai — chay `python tools/train_srt_ctc.py` truoc."
                )
            d = load_srt(path, self.vocab_enc)
            print(f"Predicted SRT for {split}: {len(d)} samples  ({path})")
            return d
        path = os.path.join(self.online_dir, f"{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} khong ton tai — chay `python tools/prep_online.py` truoc."
            )
        d = load_online(path)
        print(f"Online trajectories for {split}: {len(d)} samples")
        return d

    def _stroke_labels(self, split: str) -> dict:
        """Auxiliary per-stroke symbol labels; training split only, optional."""
        path = os.path.join(self.online_dir, f"stroke_labels_{split}.npz")
        if not os.path.exists(path):
            return {}
        d = load_stroke_labels(path)
        print(f"Stroke symbol labels for {split}: {len(d)} samples")
        return d

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = build_dataset(
                    archive, self._traj("train"), "train", self.vocab_dec,
                    self.batch_size, self._stroke_labels("train"))
                self.val_dataset = build_dataset(archive, self._traj(self.test_year), self.test_year, self.vocab_dec, 1)
            if stage == "test" or stage is None:
                self.test_dataset = build_dataset(archive, self._traj(self.test_year), self.test_year, self.vocab_dec, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            # distortion on the training split only
            collate_fn=partial(collate_fn, augment=True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    batch_size = 2

    parser = ArgumentParser()
    parser = CROHMEDatamodule.add_argparse_args(parser)

    args = parser.parse_args(["--batch_size", f"{batch_size}"])

    dm = CROHMEDatamodule(**vars(args))
    dm.setup()

    train_loader = dm.train_dataloader()
