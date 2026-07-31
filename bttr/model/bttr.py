from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from bttr.utils import Hypothesis

from .decoder import Decoder
from .encoder_img import ImgEncoder
from .encoder_seq import SeqEncoder


class BTTR(pl.LightningModule):
    def __init__(
        self,
        vocab_size_dec: int,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        fusion: str = "dual_shared",
        bidirectional: bool = True,
        traj_dim: int = 8,
        traj_downsample: int = 4,
        traj_encoder: str = "transformer",   # "transformer" | "gru" (TAP-style)
        traj_stroke_pooling: bool = False,   # stroke-level units, SCAN-style [22]
        traj_aux_classes: int = 0,           # >0 enables the per-stroke symbol head
    ):
        super().__init__()

        self.fusion = fusion
        self.bidirectional = bidirectional

        self.encoder_img = ImgEncoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )

        self.encoder_seq = SeqEncoder(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            in_dim=traj_dim,
            downsample=traj_downsample,
            encoder_type=traj_encoder,
            stroke_pooling=traj_stroke_pooling,
            aux_num_classes=traj_aux_classes,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size_dec,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fusion=fusion,
            bidirectional=bidirectional,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, traj: FloatTensor, traj_mask: LongTensor,
        tgt: LongTensor, return_aux: bool = False
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        traj : FloatTensor
            [b, l1, 8] online point features
        traj_mask : BoolTensor
            [b, l1], True marks padding
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature_offline, img_feature_mask = self.encoder_img(img, img_mask)  # [b, t, d]
        # the online encoder shortens its input, so it returns the matching mask
        traj_mask_in = traj_mask
        feature_online, traj_mask = self.encoder_seq(traj, traj_mask)  # [b, l1', d]

        # computed before the bidirectional duplication below, so the batch still
        # lines up one to one with the stroke labels
        aux_logits = (
            self.encoder_seq.stroke_logits(feature_online, traj_mask, traj, traj_mask_in)
            if return_aux else None
        )

        if self.fusion == "concat":
            # early fusion: concatenate the two memories (and their masks) along the token axis
            feature_offline = torch.cat((feature_offline, feature_online), dim=1)
            img_feature_mask = torch.cat((img_feature_mask, traj_mask), dim=1)

        if self.bidirectional:
            # duplicate features to match the bidirectional (2b) target batch
            feature_offline = torch.cat((feature_offline, feature_offline), dim=0)
            feature_online = torch.cat((feature_online, feature_online), dim=0)
            img_feature_mask = torch.cat((img_feature_mask, img_feature_mask), dim=0)
            traj_mask = torch.cat((traj_mask, traj_mask), dim=0)

        out = self.decoder(feature_offline, feature_online, img_feature_mask, traj_mask, tgt)

        return (out, aux_logits) if return_aux else out

    def beam_search(
        self, img: FloatTensor, img_mask: LongTensor, traj: FloatTensor, traj_mask: LongTensor,
        beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [1, 1, h', w']
        img_mask: LongTensor
            [1, h', w']
        traj : FloatTensor
            [1, l1, 8]
        traj_mask : BoolTensor
            [1, l1]
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature_offline, img_feature_mask = self.encoder_img(img, img_mask)  # [b, t, d]

        feature_online, traj_mask = self.encoder_seq(traj, traj_mask)

        if self.fusion == "concat":
            feature_offline = torch.cat((feature_offline, feature_online), dim=1)
            img_feature_mask = torch.cat((img_feature_mask, traj_mask), dim=1)

        return self.decoder.beam_search(feature_offline, feature_online, img_feature_mask, traj_mask, beam_size, max_len)
