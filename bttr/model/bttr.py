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
        vocab_size_enc: int,
        vocab_size_dec: int,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder_img = ImgEncoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )

        self.encoder_seq = SeqEncoder(
            vocab_size=vocab_size_enc,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size_dec,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

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
        feature_offline, img_feature_mask = self.encoder_img(img, img_mask)  # [b, t, d]
        feature_offline = torch.cat((feature_offline, feature_offline), dim=0)  # [2b, t, d]

        feature_online = self.encoder_seq(sequence_feature, sequence_feature_mask) 
        feature_online = torch.cat((feature_online, feature_online), dim=0)
        sequence_feature_mask = torch.cat((sequence_feature_mask, sequence_feature_mask), 0)

        img_feature_mask = torch.cat((img_feature_mask, img_feature_mask), dim=0)

        out = self.decoder(feature_offline, feature_online, img_feature_mask, sequence_feature_mask, tgt)

        return out

    def beam_search(
        self, img: FloatTensor, img_mask: LongTensor, sequence_feature:FloatTensor, sequence_feature_mask: LongTensor,
        beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [1, 1, h', w']
        img_mask: LongTensor
            [1, h', w']
        
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature_offline, img_feature_mask = self.encoder_img(img, img_mask)  # [b, t, d]

        feature_online = self.encoder_seq(sequence_feature, sequence_feature_mask) 

        return self.decoder.beam_search(feature_offline, feature_online, img_feature_mask, sequence_feature_mask, beam_size, max_len)
