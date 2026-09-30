"""Smoke tests for the online-trajectory branch.

Run inside the `bttr` conda env, from the project root:

    conda run -n bttr --no-capture-output python test_component.py

Covers step 2 of KE_HOACH_ONLINE_TRAJECTORY.md: shapes and padding masks of the
new SeqEncoder, a full BTTR forward for every fusion mode, and one real batch
coming out of the datamodule.  Everything runs on CPU in a few seconds; it does
not replace the fast_dev_run / overfit checks that follow.
"""

import torch

from bttr.model.bttr import BTTR
from bttr.model.decoder import Decoder
from bttr.model.encoder_seq import SeqEncoder

D_MODEL = 64
TRAJ_DIM = 8


def _ok(name):
    print(f"  PASS  {name}")


def test_decoder():
    decoder = Decoder(
        vocab_size=101, d_model=512, nhead=8, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.1,
    )
    src1 = torch.randn(2, 4, 512)
    src2 = torch.randn(2, 4, 512)
    src1_mask = torch.randint(0, 2, (2, 4)).bool()
    src2_mask = torch.randint(0, 2, (2, 4)).bool()
    tgt = torch.randint(0, 101, (2, 4))
    out = decoder(src1, src2, src1_mask, src2_mask, tgt)
    assert out.shape == (2, 4, 101), out.shape
    _ok("decoder forward")


def test_seq_encoder_shapes():
    """Output length and mask length must agree for every downsample factor."""
    for factor in (1, 2, 4, 8):
        enc = SeqEncoder(d_model=D_MODEL, nhead=4, num_encoder_layers=2,
                         dim_feedforward=128, dropout=0.0,
                         in_dim=TRAJ_DIM, downsample=factor).eval()
        for length in (7, 64, 313, 1011):
            traj = torch.randn(3, length, TRAJ_DIM)
            mask = torch.zeros(3, length, dtype=torch.bool)
            with torch.no_grad():
                mem, out_mask = enc(traj, mask)
            assert mem.shape[0] == 3 and mem.shape[2] == D_MODEL, mem.shape
            assert out_mask.shape == mem.shape[:2], (out_mask.shape, mem.shape)
            expected = -(-length // factor)  # ceil
            assert mem.shape[1] == expected, (factor, length, mem.shape[1], expected)
    _ok("SeqEncoder output length and mask length")


def test_seq_encoder_mask():
    """A sample must encode the same however much padding shares its batch.

    This is the check that catches a wrongly shrunk mask, or padding bleeding
    through the convolution -- failure modes that still train and still
    converge, just worse.
    """
    for enc_type in ("transformer", "gru"):
        _padding_invariance_for(enc_type)


def _padding_invariance_for(enc_type: str):
    torch.manual_seed(0)
    enc = SeqEncoder(d_model=D_MODEL, nhead=4, num_encoder_layers=2,
                     dim_feedforward=128, dropout=0.0,
                     in_dim=TRAJ_DIM, downsample=4, encoder_type=enc_type).eval()

    real_len = 200
    x = torch.randn(1, real_len, TRAJ_DIM)

    def run(pad_len):
        padded = torch.zeros(1, pad_len, TRAJ_DIM)
        padded[0, :real_len] = x[0]
        mask = torch.ones(1, pad_len, dtype=torch.bool)
        mask[0, :real_len] = False
        with torch.no_grad():
            return enc(padded, mask)

    with torch.no_grad():
        mem_a, mask_a = enc(x, torch.zeros(1, real_len, dtype=torch.bool))
    mem_b, mask_b = run(600)
    mem_c, mask_c = run(901)

    valid = mem_a.shape[1]
    assert (~mask_a).sum().item() == valid
    assert (~mask_b).sum().item() == valid, ((~mask_b).sum().item(), valid)
    assert (~mask_c).sum().item() == valid, ((~mask_c).sum().item(), valid)

    d_ab = (mem_a[0, :valid] - mem_b[0, :valid]).abs().max().item()
    d_bc = (mem_b[0, :valid] - mem_c[0, :valid]).abs().max().item()
    assert d_ab < 1e-5, f"padded vs unpadded: {d_ab}"
    assert d_bc < 1e-5, f"pad 600 vs pad 901: {d_bc}"
    _ok(f"SeqEncoder padding invariance, {enc_type} (max diff {max(d_ab, d_bc):.2e})")


def test_bttr_all_fusions():
    tgt = torch.randint(0, 101, (4, 6))
    img = torch.randn(2, 1, 128, 256)
    img_mask = torch.zeros(2, 128, 256, dtype=torch.bool)
    traj = torch.randn(2, 313, TRAJ_DIM)
    traj_mask = torch.zeros(2, 313, dtype=torch.bool)
    traj_mask[1, 250:] = True  # second sample is shorter

    for fusion in ("dual_shared", "offline", "online", "concat", "cascaded"):
        model = BTTR(
            vocab_size_dec=101, d_model=D_MODEL, growth_rate=8, num_layers=4,
            nhead=4, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=128, dropout=0.0, fusion=fusion, bidirectional=True,
            traj_dim=TRAJ_DIM, traj_downsample=4,
        ).eval()
        with torch.no_grad():
            out = model(img, img_mask, traj, traj_mask, tgt)
        assert out.shape == (4, 6, 101), (fusion, out.shape)
        _ok(f"BTTR forward, fusion={fusion}")


def test_backward():
    """Gradients must reach the new input projection, not just the decoder."""
    model = BTTR(
        vocab_size_dec=101, d_model=D_MODEL, growth_rate=8, num_layers=4,
        nhead=4, num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=128, dropout=0.0, fusion="dual_shared", bidirectional=True,
        traj_dim=TRAJ_DIM, traj_downsample=4,
    )
    out = model(
        torch.randn(2, 1, 128, 256),
        torch.zeros(2, 128, 256, dtype=torch.bool),
        torch.randn(2, 313, TRAJ_DIM),
        torch.zeros(2, 313, dtype=torch.bool),
        torch.randint(0, 101, (4, 6)),
    )
    out.sum().backward()
    g = model.encoder_seq.point_proj[0].weight.grad
    assert g is not None and g.abs().sum() > 0, "no gradient into point_proj"
    for i, stage in enumerate(model.encoder_seq.reduce):
        # every convolution of every stage, not just the strided one
        for name in ("conv_a", "conv_b", "conv_down"):
            g = getattr(stage, name).weight.grad
            assert g is not None and g.abs().sum() > 0, \
                f"no gradient into {name} of conv stage {i}"
    _ok("gradients reach point_proj and the conv front end")


def test_datamodule_batch():
    """One real batch: shapes, dtypes, and the mask lining up with the data."""
    from bttr.datamodule import CROHMEDatamodule

    dm = CROHMEDatamodule(test_year="2014", batch_size=8, num_workers=0)
    dm.setup(stage="test")
    batch = next(iter(dm.test_dataloader()))

    assert batch.traj.dim() == 3 and batch.traj.shape[2] == TRAJ_DIM, batch.traj.shape
    assert batch.traj_mask.shape == batch.traj.shape[:2]
    assert batch.traj_mask.dtype == torch.bool
    assert batch.traj.dtype == torch.float32
    # every sample must keep at least one valid point
    assert ((~batch.traj_mask).sum(dim=1) > 0).all()
    # padded positions must actually be zero (a batch of one has no padding)
    padded = batch.traj[batch.traj_mask]
    assert padded.numel() == 0 or padded.abs().max().item() == 0.0
    # y is normalised to [0, 1] by construction
    y = batch.traj[..., 1][~batch.traj_mask]
    assert 0.0 <= y.min().item() and y.max().item() <= 1.0 + 1e-3, (y.min(), y.max())
    _ok(f"datamodule batch {tuple(batch.traj.shape)}, {len(batch)} samples")


if __name__ == "__main__":
    print("== model ==")
    test_decoder()
    test_seq_encoder_shapes()
    test_seq_encoder_mask()
    test_bttr_all_fusions()
    test_backward()
    print("== data ==")
    test_datamodule_batch()
    print("\nTAT CA PASS")
