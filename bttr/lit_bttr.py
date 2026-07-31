import zipfile

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor

from bttr.datamodule import Batch
from bttr.datamodule.vocab import CROHMEVocab
from bttr.model.bttr import BTTR
from bttr.utils import ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out, to_tgt_output
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
        # ablation switches
        fusion: str = "dual_shared",      # dual_shared | offline | online | concat | cascaded
        bidirectional: bool = True,       # True = L2R+R2L (Ours); False = L2R only
        traj_dim: int = 8,                # online point feature width (TAP-style)
        traj_downsample: int = 4,         # length reduction inside the online encoder
        traj_encoder: str = "transformer",  # "transformer" | "gru" (TAP-style [33])
        traj_stroke_pooling: bool = False,  # stroke-level units, SCAN-style [22]
        aux_stroke_weight: float = 0.0,     # >0 adds the per-stroke symbol loss
        vocab_dec: str = "vocab/dictionary.txt",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_dec = CROHMEVocab(vocab_dec)

        self.bttr = BTTR(
            vocab_size_dec=len(self.vocab_dec),
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fusion=fusion,
            bidirectional=bidirectional,
            traj_dim=traj_dim,
            traj_downsample=traj_downsample,
            traj_encoder=traj_encoder,
            traj_stroke_pooling=traj_stroke_pooling,
            traj_aux_classes=len(self.vocab_dec) if aux_stroke_weight > 0 else 0,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, traj: FloatTensor, traj_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        traj : FloatTensor
            [b, l1, 8]
        traj_mask : BoolTensor
            [b, l1]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.bttr(img, img_mask, traj, traj_mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        traj: FloatTensor,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> str:
        """for inference, one image at a time

        Parameters
        ----------
        img : FloatTensor
            [1, h, w]
        traj : FloatTensor
            [1, l1, 8] online point features, unpadded
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
        img_mask = torch.zeros_like(img, dtype=torch.bool)  # squeeze channel
        # a single unpadded trajectory: the mask is [1, l1], not the shape of traj
        traj_mask = torch.zeros(traj.shape[:2], dtype=torch.bool, device=traj.device)
        hyps = self.bttr.beam_search(img.unsqueeze(0), img_mask, traj, traj_mask, beam_size, max_len)
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
        return self.vocab_dec.indices2label(best_hyp.seq)

    def training_step(self, batch: Batch, _):
        if self.hparams.bidirectional:
            tgt, out = to_bi_tgt_out(batch.indices, self.device)
        else:
            tgt, out = to_tgt_output(batch.indices, "l2r", self.device)
        use_aux = self.hparams.aux_stroke_weight > 0 and batch.stroke_labels is not None
        if use_aux:
            out_hat, aux_logits = self.bttr(
                batch.imgs, batch.mask, batch.traj, batch.traj_mask, tgt, return_aux=True
            )
        else:
            out_hat = self(batch.imgs, batch.mask, batch.traj, batch.traj_mask, tgt)
        loss = ce_loss(out_hat, out, self.vocab_dec.PAD_IDX)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, )

        if use_aux:
            # one symbol prediction per stroke; -100 marks padding and the few
            # strokes whose traceGroup symbol is outside the decoder vocabulary
            lab = batch.stroke_labels[:, : aux_logits.size(1)]
            aux = F.cross_entropy(
                rearrange(aux_logits, "b s c -> (b s) c"),
                rearrange(lab, "b s -> (b s)"),
                ignore_index=-100,
            )
            self.log("train_aux", aux, on_step=False, on_epoch=True, sync_dist=True)
            loss = loss + self.hparams.aux_stroke_weight * aux

        return loss

    def validation_step(self, batch: Batch, _):
        if self.hparams.bidirectional:
            tgt, out = to_bi_tgt_out(batch.indices, self.device)
        else:
            tgt, out = to_tgt_output(batch.indices, "l2r", self.device)
        out_hat = self(batch.imgs, batch.mask, batch.traj, batch.traj_mask, tgt)

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
            batch.imgs, batch.mask, batch.traj, batch.traj_mask,
            self.hparams.beam_size, self.hparams.max_len
        )

        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** self.hparams.alpha))
        self.exprate_recorder(best_hyp.seq, batch.indices[0])

        return batch.img_bases[0], self.vocab_dec.indices2label(best_hyp.seq)

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
            # Deliberately "max" against monitor="val_loss": a falling loss never
            # sets a new best, so this acts as a step decay that cuts the lr every
            # `patience` checks (epoch ~24 and ~46 of a 50-epoch run). Every
            # published number in the thesis was trained under it -- switching to
            # "min" cost the offline branch 12 ExpRate points (47.61 -> 35.63).
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
