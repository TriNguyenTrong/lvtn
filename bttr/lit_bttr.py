import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from bttr.datamodule import Batch
from bttr.datamodule.vocab import CROHMEVocab
from bttr.model.bttr import BTTR
from bttr.utils import ExpRateRecorder, Hypothesis, ce_loss
from einops import rearrange, repeat



class LitBTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        num_encoder_layers: int, # seq encoder
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        # training
        learning_rate: float,
        patience: int,
        vocab_enc: str = "vocab/crohme_seq_vocab.txt",
        vocab_dec: str = "vocab/dictionary.txt",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_enc = CROHMEVocab(vocab_enc)
        self.vocab_dec = CROHMEVocab(vocab_dec)

        self.bttr = BTTR(
            vocab_size_enc=len(self.vocab_enc),
            vocab_size_dec=len(self.vocab_dec),
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, sequence_feature, sequence_feature_mask, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.bttr(img, img_mask,sequence_feature, sequence_feature_mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> str:
        """for inference, one image at a time

        Parameters
        ----------
        img : FloatTensor
            [1, h, w]
        beam_size : int, optional
            by default 10
        max_len : int, optional
            by default 200
        alpha : float, optional
            by default 1.0

        Returns
        -------
        str
            LaTex string
        """
        # assert img.dim() == 3
        # img1 = img.to(torch.int64)
        # assert False, f'test {img1.dtype}'

        # img_mask = torch.zeros_like(img1, dtype=torch.long)  # squeeze channel
        img_mask = torch.zeros_like(img, dtype=torch.bool)  # squeeze channel
        # img_mask = torch.zeros_like(img, dtype=torch.float)  # squeeze channel
        hyps = self.bttr.beam_search(img.unsqueeze(0), img_mask, beam_size, max_len)
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
        return vocab.indices2label(best_hyp.seq)

    def training_step(self, batch: Batch, _):
        tgt, out = self.vocab_dec.to_bi_tgt_out(batch.indices, self.device)
        seq, seq_mask = self.vocab_enc.to_src(batch.seq_indices, self.device)
        out_hat = self(batch.imgs, batch.mask, seq, seq_mask, tgt)
        loss = ce_loss(out_hat, out, self.vocab_dec.PAD_IDX)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, )

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = self.vocab_dec.to_bi_tgt_out(batch.indices, self.device)
        seq, seq_mask = self.vocab_enc.to_src(batch.seq_indices, self.device)
        out_hat = self(batch.imgs, batch.mask, seq, seq_mask, tgt)

        loss = ce_loss(out_hat, out, self.vocab_dec.PAD_IDX)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
    
        # hyps = self.bttr.beam_search(
        #     batch.imgs, batch.mask, self.hparams.beam_size, self.hparams.max_len
        # )
        # best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))

        # self.exprate_recorder(best_hyp.seq, batch.indices[0])
        # self.log(
        #     "val_ExpRate",
        #     self.exprate_recorder,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        # )

    def test_step(self, batch: Batch, _):
        hyps = self.bttr.beam_search(
            batch.imgs, batch.mask, self.hparams.beam_size, self.hparams.max_len
        )
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))
        self.exprate_recorder(best_hyp.seq, batch.indices[0])

        return batch.img_bases[0], vocab.indices2label(best_hyp.seq)

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"ExpRate: {exprate}")

        print(f"length of total file: {len(test_outputs)}")
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_base, pred in test_outputs:
                content = f"%{img_base}\n${pred}$".encode()
                with zip_f.open(f"{img_base}.txt", "w") as f:
                    f.write(content)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
