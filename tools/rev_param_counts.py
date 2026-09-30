"""Count parameters per block of the main system, plus the CTC recognisers.

    python tools/rev_param_counts.py
"""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CKPT = "lightning_logs/rev_long_shared/lightning_logs/version_0/checkpoints/epoch=105-step=114904-val_ExpRate=0.8054.ckpt"

def count(model):
    return sum(p.numel() for p in model.parameters())

def main():
    from bttr.lit_bttr import LitBTTR
    m = LitBTTR.load_from_checkpoint(os.path.join(ROOT, CKPT), strict=True)
    bttr = m.bttr
    lines = [f"Parameter counts -- checkpoint {CKPT}", ""]
    lines.append(f"encoder_img (image, DenseNet):        {count(bttr.encoder_img):>10,}")
    seq = bttr.encoder_seq
    lines.append(f"encoder_seq total (trajectory branch): {count(seq):>10,}")
    lines.append(f"  point_proj:                          {count(seq.point_proj):>10,}")
    lines.append(f"  reduce (downsample conv stages):     {sum(count(x) for x in seq.reduce):>10,}")
    lines.append(f"  rnn (BiGRU):                         {count(seq.rnn):>10,}")
    lines.append(f"  rnn_norm:                            {count(seq.rnn_norm):>10,}")
    if seq.aux_head is not None:
        lines.append(f"  aux_head (per-stroke classifier):    {count(seq.aux_head):>10,}")
    lines.append(f"decoder total:                          {count(bttr.decoder):>10,}")
    lines.append(f"  word_embed:                           {count(bttr.decoder.word_embed):>10,}")
    lines.append(f"  pos_enc:                              {count(bttr.decoder.pos_enc):>10,}")
    lines.append(f"  transformer layers:                   {count(bttr.decoder.model):>10,}")
    lines.append(f"  proj (output vocab):                  {count(bttr.decoder.proj):>10,}")
    lines.append(f"TOTAL model (bttr):                     {count(bttr):>10,}")
    lines.append("")

    lines.append("CTC recognisers (srt_ckpt/, online/srt_ctc*.pt):")
    from train_srt_ctc import BiLSTMCTC, Vocab
    vocab = Vocab(os.path.join(ROOT, "vocab", "crohme_seq_vocab.txt"))
    for name, ckpt, feat, hidden, layers in [
        ("srt_ckpt (Kaggle, 4-D)", "srt_ckpt/epoch17-val_wer0.1368.ckpt", 4, 128, 3),
        ("online/srt_ctc.pt (ours, 4-D)", "online/srt_ctc.pt", 4, 128, 3),
        ("online/srt_ctc_tap8.pt (ours, 8-D)", "online/srt_ctc_tap8.pt", 8, 256, 4),
        ("online/srt_ctc_tap8n.pt (ours, 8-D norm.)", "online/srt_ctc_tap8n.pt", 8, 256, 4),
    ]:
        model = BiLSTMCTC(vocab.n_classes, input_size=feat, hidden=hidden, layers=layers)
        lines.append(f"  {name}: {count(model):>10,}")

    open(os.path.join(ROOT, "results", "rev_param_counts.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
