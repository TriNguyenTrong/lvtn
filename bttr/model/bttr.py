from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from bttr.utils import Hypothesis

from .decoder import Decoder
from .decoder import Decoder1
from .encoder import Encoder
from bttr.datamodule import vocab, vocab_size


class BTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.encoder2 = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
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
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]

        feature2, mask2 = self.encoder2(sequence_feature, sequence_feature_mask) 

        mask = torch.cat((mask, mask), dim=0)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        out = self.decoder(feature, feature2, mask, mask2, tgt,tgt_pad_mask)

        return out

    def beam_search(
        self, img: LongTensor, img_mask: LongTensor, sequence_feature, sequence_feature_mask, beam_size: int, max_len: int
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
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        feature2, mask2 = self.encoder2(sequence_feature, sequence_feature_mask)  
        return self.decoder.beam_search(feature, feature2, mask, mask2, beam_size, max_len)
