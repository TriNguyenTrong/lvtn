import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import LayerNorm
from torch.nn.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .pos_enc import WordPosEnc


class SeqEncoder(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def forward(self, src: Tensor, src_padding_mask: Tensor):
        src_emb = self.pos_enc(self.word_embed(src))
        src_emb = rearrange(src_emb, "b l d -> l b d")
        memory = self.encoder(
            src_emb, mask=None, src_key_padding_mask=src_padding_mask
        )
        memory = rearrange(memory, "l b d -> b l d")
        return memory
