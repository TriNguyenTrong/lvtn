from bttr.model.decoder import Decoder, TransformerDecoderLayerMulti
import torch.nn as nn
from bttr.model.bttr import BTTR

import pytorch_lightning as pl
import torch

def test_decoder():
    decoder = Decoder(
        vocab_size=101,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )

    src1 = torch.randn(2, 4, 512)
    src2 = torch.randn(2, 4, 512)
    src1_mask = torch.randint(0, 2, (2, 4)).bool()
    src2_mask = torch.randint(0, 2, (2, 4)).bool()
    tgt = torch.randint(0, 101, (2, 4))

    out = decoder(src1, src2, src1_mask, src2_mask, tgt)
    print(out.shape)

def test_BTTR():
    model = BTTR(
        vocab_size_enc=121,
        vocab_size_dec=101,
        d_model=512,
        growth_rate=32,
        num_layers=4,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )
    
    #  img: FloatTensor, img_mask: LongTensor, sequence_feature, sequence_feature_mask, tgt: LongTensor
    img = torch.randn(2, 1, 256, 256)
    img_mask = torch.randint(0, 2, (2, 256, 256)).bool()
    sequence_feature = torch.randint(0, 121, (2, 20))
    sequence_feature_mask = torch.randint(0, 2, (2, 20)).bool()
    tgt = torch.randint(0, 101, (4, 4))
    out = model(img, img_mask, sequence_feature, sequence_feature_mask, tgt)
    print(out.shape)
    
# test_decoder()
test_BTTR()
