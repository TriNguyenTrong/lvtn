import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from .vocab import CROHMEVocab

# vocab = CROHMEVocab()

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    seq_batch = []
    label_batch = []
    feature_total = []
    seq_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, seq, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = transforms.ToTensor()(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[1]} x {fea.shape[2]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                seq_total.append(seq_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                seq_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                seq_batch.append(seq)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                seq_batch.append(seq)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    seq_total.append(seq_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, seq_total, label_total))


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


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    seq_indices: List[List[int]]  # [b, l1]
    indices: List[List[int]]  # [b, l2]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            seq_indices = self.seq_indices,
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_x = batch[2]
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

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_x, seqs_y)


def build_dataset(archive, seq_dict, folder: str, vocab_enc, vocab_dec, batch_size: int):
    data = extract_data(archive, folder)
    data = [(fname, img, vocab_enc.words2indices(seq_dict[fname].split()), vocab_dec.words2indices(formula)) 
    for fname, img, formula in data if fname in seq_dict]
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        seq_annotation: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../crohme_all.txt",
        vocab_enc: str = "vocab/crohme_seq_vocab.txt",
        vocab_dec: str = "vocab/dictionary.txt",
        test_year: str = "2014",
        batch_size: int = 8,
        num_workers: int = 5,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.seq_annotation = seq_annotation
        self.test_year = test_year
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_enc = CROHMEVocab(vocab_enc)
        self.vocab_dec = CROHMEVocab(vocab_dec)

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        seq_dict = {os.path.splitext(os.path.basename(line.strip().split('\t')[0]))[0]: line.strip().split('\t')[1] 
        for line in open(self.seq_annotation).readlines() if len(line.strip().split('\t')) == 2}
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = build_dataset(archive, seq_dict, "train", self.vocab_enc, self.vocab_dec, self.batch_size)
                self.val_dataset = build_dataset(archive, seq_dict, self.test_year, self.vocab_enc, self.vocab_dec, 1)
            if stage == "test" or stage is None:
                self.test_dataset = build_dataset(archive, seq_dict, self.test_year, self.vocab_enc, self.vocab_dec, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
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
