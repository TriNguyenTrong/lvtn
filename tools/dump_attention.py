"""Dump cross-attention weights of the LAST decoder layer, both memories, for a few samples.

No model class is edited: the last TransformerDecoderLayerMulti's `_mha_block` is monkeypatched
at the INSTANCE level (types.MethodType) to call self.multihead_attn with need_weights=True and
restored right after, exactly reproducing nn.TransformerDecoderLayer._mha_block otherwise. Because
`_cross2` (decoder.py) calls this same `_mha_block` when `separate_cross` is off (the main system's
setting), one patch captures both cross-attentions; the two calls are told apart by object identity
of the memory tensor passed in (memory1 = offline/image, memory2 = online/trajectory).

Attention is over the model's OWN prediction: beam search first finds the best hypothesis (as at
test time, bidirectional beam 10 with cross-rescoring); Hypothesis.seq is always stored left-to-
right regardless of which beam found it, so one L2R teacher-forced pass then recovers every query
position's attention weights in a single forward call.

Output: results/attn/<sample>.npz with
  attn_offline [heads, q_len, k_len_offline], attn_online [heads, q_len, k_len_online] (q_len =
  len(tokens)+1, position 0 = the query for the SOS start token), tokens (the predicted sequence,
  decoded), ground_truth, correct (bool).

    python tools/dump_attention.py
    python tools/dump_attention.py --ckpt <path> --samples name1,name2
"""
import argparse
import os
import sys
import types
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_CKPT = os.path.join(
    "lightning_logs", "abl_dual_shared_aux", "lightning_logs", "version_0",
    "checkpoints", "epoch=39-step=45560-val_loss=0.3622.ckpt")
DEFAULT_SAMPLES = ["UN19_1010_em_139", "UN_105_em_111", "UN_104_em_88"]


def find_year(name):
    with zipfile.ZipFile(os.path.join(ROOT, "data.zip")) as z:
        for year in ("2014", "2016", "2019"):
            caps = z.read(f"{year}/caption.txt").decode().splitlines()
            for line in caps:
                t = line.split()
                if t and t[0] == name:
                    return year, " ".join(t[1:])
    raise KeyError(f"{name}: not found in any test split's caption.txt")


def load_sample(name, year):
    from bttr.datamodule.datamodule import load_online
    with zipfile.ZipFile(os.path.join(ROOT, "data.zip")) as z:
        img = Image.open(z.open(f"{year}/{name}.bmp")).copy()
    img_t = transforms.ToTensor()(img).unsqueeze(0)                # [1, 1, H, W]
    img_mask = torch.zeros((1,) + img_t.shape[2:], dtype=torch.bool)
    traj_dict = load_online(os.path.join(ROOT, "online", f"{year}.npz"))
    traj = torch.from_numpy(traj_dict[name]).unsqueeze(0)           # [1, l, 8]
    traj_mask = torch.zeros((1, traj.shape[1]), dtype=torch.bool)
    return img_t, img_mask, traj, traj_mask


def patched_mha_block(store):
    """Bound-method replacement for one TransformerDecoderLayerMulti instance's inherited
    nn.TransformerDecoderLayer._mha_block: identical except need_weights=True, and the
    (detached) weights are appended to `store` in call order. `Decoder.forward` rearranges
    memory1/memory2 into new tensors before the decoder stack runs, so telling the two calls
    apart by object identity does not work; call order does, since `forward()`
    (decoder.py's TransformerDecoderLayerMulti.forward) always issues the memory1 cross-
    attention before memory2's, once each, for every fusion mode that uses both."""
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        out, w = self.multihead_attn(x, mem, mem, attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=True, average_attn_weights=False)
        store.append(w.detach()[0].cpu().numpy())  # [heads, q_len, k_len], batch=1
        return self.dropout2(out)
    return _mha_block


def run_one(model, name, year, gt):
    img, img_mask, traj, traj_mask = load_sample(name, year)
    device = next(model.parameters()).device
    img, img_mask, traj, traj_mask = (t.to(device) for t in (img, img_mask, traj, traj_mask))
    bttr = model.bttr

    with torch.no_grad():
        hyps = bttr.beam_search(img, img_mask, traj, traj_mask,
                                model.hparams.beam_size, model.hparams.max_len)
    best = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
    pred = model.vocab_dec.indices2label(best.seq)

    with torch.no_grad():
        feat_off, mask_off = bttr.encoder_img(img, img_mask)
        feat_on, mask_on = bttr.encoder_seq(traj, traj_mask)

    # Hypothesis.seq is already stored left-to-right regardless of which beam (L2R or R2L) found
    # it (bttr/utils.py reverses R2L sequences back on construction), so the teacher-forced pass
    # that recovers the attention is always run L2R: tgt_in = [SOS, tok_1, ..., tok_L].
    tgt_in = torch.tensor([model.vocab_dec.SOS_IDX] + best.seq, device=device).unsqueeze(0)

    last_layer = bttr.decoder.model.layers[-1]
    store = []
    original = last_layer._mha_block
    last_layer._mha_block = types.MethodType(patched_mha_block(store), last_layer)
    try:
        with torch.no_grad():
            bttr.decoder(feat_off, feat_on, mask_off, mask_on, tgt_in)
    finally:
        last_layer._mha_block = original  # restore even on error; instance is not left patched

    # order: memory1 (offline) is always queried before memory2 (online) when both are used;
    # "online"-only fusion queries memory2 alone, so a single capture belongs to it instead
    if len(store) == 2:
        attn_offline, attn_online = store
    elif bttr.fusion == "online":
        attn_offline, attn_online = np.zeros((0,)), store[0]
    else:
        attn_offline, attn_online = store[0], np.zeros((0,))

    out = dict(
        attn_offline=attn_offline,
        attn_online=attn_online,
        tokens=np.array(model.vocab_dec.indices2words(best.seq)),
        decoding_order="l2r (teacher-forced on the model's own bidirectional-beam output)",
        prediction=pred,
        ground_truth=gt,
        correct=pred == gt,
        fusion=bttr.fusion,
        separate_cross=getattr(last_layer, "separate_cross", False),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "attn"))
    args = ap.parse_args()

    from bttr.lit_bttr import LitBTTR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitBTTR.load_from_checkpoint(os.path.join(ROOT, args.ckpt), strict=True).eval().to(device)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in args.samples.split(","):
        year, gt = find_year(name)
        out = run_one(model, name, year, gt)
        path = os.path.join(args.out_dir, f"{name}.npz")
        np.savez(path, **out)
        flag = "OK" if out["correct"] else "WRONG"
        print(f"{name} (test {year}, {flag}): "
              f"offline {out['attn_offline'].shape}, online {out['attn_online'].shape} -> {path}")


if __name__ == "__main__":
    main()
