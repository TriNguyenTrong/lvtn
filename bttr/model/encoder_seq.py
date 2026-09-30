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


# Per-feature statistics of online/train.npz (all 2,754,815 training points).
# The six continuous channels are standardised; the two pen bits are already in
# [0, 1] and are passed through, so standardising them would only amplify the
# rare pen-up spike.  Without this, x (std 3.02) outweighs dx (std 0.107) by 28x
# going into a single Linear, and the stroke-shape channels contribute a few
# percent of the activation while absolute position -- the one feature that lets
# the model identify a training sample outright -- dominates.
TRAJ_FEAT_MEAN = (2.830163, 0.510638, 0.011059, 0.001062, 0.022132, 0.002129, 0.0, 0.0)
TRAJ_FEAT_STD = (3.018715, 0.245498, 0.107065, 0.069360, 0.152776, 0.101755, 1.0, 1.0)


class _ChannelNorm(nn.Module):
    """LayerNorm over channels at each position of a [b, d, l] tensor.

    Deliberately not BatchNorm or GroupNorm: both pool along the time axis (or
    across the batch), so a sample's encoding would depend on what is padded
    next to it, breaking the padding invariance test_component.py checks to 1e-5.
    Normalising per position keeps every position independent.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _ReduceStage(nn.Module):
    """Two residual convolutions at full rate, then one strided convolution.

    Three layers per halving instead of one.  With kernel 5 the old two-layer
    front end saw 13 points per output, about 0.39 of the height-normalised arc
    length -- a third of a symbol, so no memory vector ever covered a whole one.
    Stacking to six layers over two stages widens that to 37 points (~1.1), which
    does cover a symbol.
    """

    def __init__(self, d_model: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.conv_a = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.norm_a = _ChannelNorm(d_model)
        self.conv_b = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.norm_b = _ChannelNorm(d_model)
        self.conv_down = nn.Conv1d(d_model, d_model, kernel_size=5, stride=stride, padding=2)
        self.norm_down = _ChannelNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor):
        """x: [b, d, l], mask: [b, l] with True marking padding."""
        # Re-mask after every convolution, not merely once per stage: each kernel
        # reaches back into real data, so the first padded position of any conv
        # output is non-zero, and the next layer would mix that into the last
        # real position.
        h = self.conv_a(x)
        h = F.relu(self.norm_a(h), inplace=True).masked_fill(mask.unsqueeze(1), 0.0)
        h = self.conv_b(h)
        h = F.relu(self.norm_b(h), inplace=True).masked_fill(mask.unsqueeze(1), 0.0)
        x = x + h

        x = self.conv_down(x)
        if self.stride > 1:
            mask = SeqEncoder._shrink_mask(mask, x.size(-1), self.stride)
        x = F.relu(self.norm_down(x), inplace=True).masked_fill(mask.unsqueeze(1), 0.0)
        return x, mask


def _stroke_pool(feat: Tensor, pen_up: Tensor, pad_mask: Tensor):
    """Average the point features of each pen stroke into one vector.

    Follows the stroke-level representation of SCAN [22]: the stroke, not the
    trace point, is the unit the decoder attends over.  Grouping the points of a
    stroke explicitly is what removes most of the symbol-segmentation burden
    from the attention, and unlike a symbol grouping it costs no annotation --
    stroke boundaries are the pen-up bit the digitiser already records, so this
    reads nothing that would not exist at test time.

    Parameters
    ----------
    feat : [b, l, d] point features, padding already zeroed
    pen_up : [b, l] 1.0 at the last point of each stroke
    pad_mask : [b, l] True marks padding

    Returns
    -------
    [b, s, d] stroke features and their [b, s] padding mask, s = max stroke count
    """
    stroke_id, n_strokes = _stroke_ids(pen_up, pad_mask)
    return _pool_by_id(feat, stroke_id, pad_mask, n_strokes)


def _stroke_ids(pen_up: Tensor, pad_mask: Tensor):
    """[b, l] stroke index per point, and [b] stroke count per sample."""
    valid = ~pad_mask
    # the point carrying pen-up still belongs to the stroke it closes, hence the
    # subtraction
    stroke_id = (torch.cumsum(pen_up, dim=1) - pen_up).long()
    stroke_id = torch.where(valid, stroke_id, torch.zeros_like(stroke_id))
    n_strokes = (stroke_id * valid).amax(dim=1) + 1
    return stroke_id, n_strokes


def _pool_by_id(feat: Tensor, stroke_id: Tensor, pad_mask: Tensor, n_strokes: Tensor):
    """Mean-pool [b, l, d] into [b, s, d] according to a per-position group id."""
    valid = ~pad_mask
    s_max = int(n_strokes.max().item())
    stroke_id = stroke_id.clamp(max=s_max - 1)

    w = valid.to(feat.dtype).unsqueeze(-1)                      # [b, l, 1]
    idx = stroke_id.unsqueeze(-1).expand(-1, -1, feat.size(-1))  # [b, l, d]

    sums = torch.zeros(feat.size(0), s_max, feat.size(-1),
                       dtype=feat.dtype, device=feat.device)
    sums.scatter_add_(1, idx, feat * w)
    counts = torch.zeros(feat.size(0), s_max, 1,
                         dtype=feat.dtype, device=feat.device)
    counts.scatter_add_(1, idx[..., :1], w)

    out = sums / counts.clamp(min=1.0)
    out_mask = torch.arange(s_max, device=feat.device).unsqueeze(0) >= n_strokes.unsqueeze(1)
    out = out.masked_fill(out_mask.unsqueeze(-1), 0.0)
    return out, out_mask


class SeqEncoder(pl.LightningModule):
    """Online branch encoder over the raw pen trajectory.

    The input is a sequence of points sampled along the strokes, each described
    by the 8-D feature of TAP (Zhang, Du & Dai, IEEE TMM 2019, eq. 2):
    (x, y, dx, dy, dx', dy', pen-down, pen-up).  Points come from
    tools/prep_online.py and carry no symbol label, so the branch sees only what
    a digitiser actually records.

    A strided convolutional front end shortens the sequence before self-attention.
    Trajectories run to a few hundred points -- an order of magnitude longer than
    the symbol sequences this encoder used to consume -- and attention cost grows
    quadratically, so the reduction is what keeps the two branches comparable in
    memory and speed.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        in_dim: int = 8,
        downsample: int = 4,
        max_len: int = 3000,
        encoder_type: str = "transformer",
        stroke_pooling: bool = False,
        aux_num_classes: int = 0,
        input_mode: str = "traj",
        vocab_size: int = 0,
    ):
        super().__init__()

        # "traj": raw pen points, 8-D TAP features (the branch built for this thesis'
        #   trajectory experiments).
        # "srt" : a symbol-relation token sequence, embedded exactly as the original
        #   encoder did. Feed it ground-truth SRT and you get the oracle variant;
        #   feed it SRT predicted from the trajectory by a separate recogniser and
        #   the system is oracle-free while the architecture stays untouched.
        assert input_mode in ("traj", "srt")
        self.input_mode = input_mode
        if input_mode == "srt":
            assert vocab_size > 0, "srt mode needs the encoder vocabulary size"
            downsample = 1
            stroke_pooling = False
            encoder_type = "transformer"   # as the original SRT encoder was
            self.word_embed = nn.Sequential(
                nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
            )

        assert downsample in (1, 2, 4, 8), "downsample must be a power of two up to 8"
        assert encoder_type in ("transformer", "gru")
        self.in_dim = in_dim
        self.downsample = downsample
        # With stroke pooling the convolutions run at stride 1 and it is the
        # pooling that shortens the sequence, so `downsample` no longer sets the
        # length -- it sets the conv DEPTH, 3 layers per power of two, keeping the
        # receptive field of the non-pooling variant for a fair comparison.
        self.stroke_pooling = stroke_pooling
        # Defaults to "transformer" so checkpoints trained before the GRU option
        # existed keep loading into the architecture they were trained as.
        self.encoder_type = encoder_type

        # Buffers so the normalisation travels with the checkpoint and train and
        # test can never disagree about it.
        self.register_buffer(
            "feat_mean", torch.tensor(TRAJ_FEAT_MEAN[:in_dim]).view(1, 1, in_dim)
        )
        self.register_buffer(
            "feat_std", torch.tensor(TRAJ_FEAT_STD[:in_dim]).view(1, 1, in_dim)
        )

        # Auxiliary head: predict the symbol each stroke belongs to.  Trained
        # from <traceGroup>, which exists only in the training InkML, and never
        # evaluated -- so this adds supervision without putting any ground truth
        # into the model's input.  The online branch otherwise learns from 8,834
        # LaTeX sequences; this gives it 121,161 labelled strokes as well.
        self.aux_num_classes = aux_num_classes
        self.aux_head = (
            nn.Linear(d_model, aux_num_classes) if aux_num_classes > 0 else None
        )

        self.point_proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model)
        )

        # A ModuleList rather than a Sequential: each stage halves the sequence,
        # so the padding mask has to be recomputed between them, see forward().
        self.reduce = nn.ModuleList(
            _ReduceStage(d_model, stride=1 if stroke_pooling else 2)
            for _ in range(int(math.log2(downsample)))
        )

        # WordPosEnc defaults to 500 positions, which was ample for symbol
        # sequences but not for trajectories: the longest training expression is
        # 2038 points, still 510 after the 4x reduction.  Sizing the table by
        # MAX_TRAJ_LEN covers downsample=1 as well and costs ~3 MB.
        if encoder_type == "transformer":
            self.pos_enc = WordPosEnc(d_model, max_len=max_len)
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            )
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )
        else:
            # Stacked bidirectional GRU, following TAP [33], the reference online
            # system that reaches 50.41% ExpRate on CROHME 2014 from the same
            # 8.8k training expressions.  With a training set that size a
            # Transformer has too little inductive bias: measured train/test
            # token accuracy was 76.1/46.0 for the Transformer encoder against
            # 96.2/84.5 for the DenseNet image branch -- it both fit the training
            # set worse and generalised worse.  Recurrence supplies the
            # sequential prior the point sequence needs.
            assert d_model % 2 == 0, "d_model must be even to split across directions"
            self.rnn = nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=num_encoder_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_encoder_layers > 1 else 0.0,
            )
            # No positional encoding here: recurrence already carries order, and
            # adding one only fights the recurrence.
            self.rnn_norm = LayerNorm(d_model)

    @staticmethod
    def _shrink_mask(mask: Tensor, out_len: int, factor: int) -> Tensor:
        """Shrink a padding mask to match the convolution output length.

        Recomputing it from the valid lengths rather than slicing avoids the
        silent failure mode where attention keeps reading padded positions: the
        model still trains and still converges, it just quietly loses accuracy.
        """
        lens = (~mask).sum(dim=1)
        new_lens = torch.div(lens + factor - 1, factor, rounding_mode="floor")
        new_lens = new_lens.clamp(min=1, max=out_len)
        ar = torch.arange(out_len, device=mask.device).unsqueeze(0)
        return ar >= new_lens.unsqueeze(1)

    def forward(self, traj: Tensor, traj_padding_mask: Tensor):
        """
        Parameters
        ----------
        traj : FloatTensor
            [b, l, in_dim] point features
        traj_padding_mask : BoolTensor
            [b, l], True marks padding

        Returns
        -------
        Tuple[FloatTensor, BoolTensor]
            memory [b, l', d] and its padding mask [b, l'], l' = ceil(l / downsample)
        """
        if self.input_mode == "srt":
            # token ids [b, l] -> embeddings; no convolutions, the sequence is
            # already short (19 tokens on average against 312 trace points)
            x = self.pos_enc(self.word_embed(traj.long()))
            x = rearrange(x, "b l d -> l b d")
            memory = self.encoder(x, mask=None, src_key_padding_mask=traj_padding_mask)
            return rearrange(memory, "l b d -> b l d"), traj_padding_mask

        x = (traj - self.feat_mean) / self.feat_std
        x = self.point_proj(x)
        # Zero the padded positions, and keep zeroing them after every conv
        # stage.  Two reasons, both about a sample encoding the same way no
        # matter what shares its batch: the projection ends in a LayerNorm, so a
        # padded all-zero point comes out as LayerNorm(bias) rather than zero;
        # and each kernel reaches back into real data, so the first padded
        # position of a conv output is non-zero even when its input was clean.
        # Re-masking makes the convolution see exactly the zeros it pads the
        # sequence end with.
        x = x.masked_fill(traj_padding_mask.unsqueeze(-1), 0.0)

        if self.downsample > 1:
            x = rearrange(x, "b l d -> b d l")
            for stage in self.reduce:
                x, traj_padding_mask = stage(x, traj_padding_mask)
            x = rearrange(x, "b d l -> b l d")

        if self.stroke_pooling:
            # pen-up is channel 7 of the raw input and is left unstandardised,
            # so it is still exactly 0/1 here; the convolutions ran at stride 1
            # so points and features are still aligned one to one.
            x, traj_padding_mask = _stroke_pool(
                x, traj[..., 7], traj_padding_mask
            )

        if self.encoder_type == "transformer":
            x = self.pos_enc(x)
            x = rearrange(x, "b l d -> l b d")
            memory = self.encoder(x, mask=None, src_key_padding_mask=traj_padding_mask)
            memory = rearrange(memory, "l b d -> b l d")
        else:
            # Pack before the RNN. Without it the backward direction would start
            # by consuming the padding, so a sample's encoding would depend on
            # what shares its batch -- the same invariance the masking above
            # protects for the convolutions.
            lengths = (~traj_padding_mask).sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.rnn(packed)
            memory, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1)
            )
            memory = self.rnn_norm(memory)
            memory = memory.masked_fill(traj_padding_mask.unsqueeze(-1), 0.0)

        return memory, traj_padding_mask

    def stroke_logits(self, memory: Tensor, memory_mask: Tensor,
                      traj: Tensor, traj_mask: Tensor) -> Tensor:
        """[b, s, aux_num_classes] -- one symbol prediction per pen stroke.

        Pools the encoder memory back to stroke units.  When the convolutions
        ran with stride the memory is shorter than the input, so the stroke ids
        are subsampled by the same factor; a stroke averages 22.7 points, about
        5.7 memory positions at downsample 4, so the alignment is coarse but
        never merges two strokes into one slot.
        """
        assert self.aux_head is not None, "encoder built without an auxiliary head"
        stroke_id, n_strokes = _stroke_ids(traj[..., 7], traj_mask)

        if not self.stroke_pooling and self.downsample > 1:
            stroke_id = stroke_id[:, :: self.downsample][:, : memory.size(1)]
        elif self.stroke_pooling:
            # memory is already one vector per stroke
            return self.aux_head(memory)

        pooled, _ = _pool_by_id(memory, stroke_id, memory_mask, n_strokes)
        return self.aux_head(pooled)
