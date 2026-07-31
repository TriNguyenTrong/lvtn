from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping


def test():
    print("vocab size:", len(vocab))
    # import torch

    # from bttr.lit_bttr import LitBTTR

    # model = LitBTTR(d_model=256,
    #                 growth_rate=24,
    #                 num_layers=16,
    #                 nhead= 8,
    #                 dim_feedforward= 1024,
    #                 dropout= 0.3,
    #                 num_decoder_layers= 3,
    #                 beam_size= 10,
    #                 max_len= 200,
    #                 alpha= 1.0,
    #                 learning_rate= 1.0,
    #                 patience= 20)
    

# test()

def parse_args():
    """Switches are command-line options so a night of ablations can be chained
    without editing this file between runs. The defaults are the configuration
    of the thesis' main system."""
    p = ArgumentParser()
    p.add_argument("--fusion", default="dual_shared",
                   choices=["dual_shared", "offline", "online", "concat", "cascaded"])
    p.add_argument("--unidirectional", action="store_true",
                   help="train L2R only instead of L2R+R2L")
    p.add_argument("--traj-encoder", default="gru", choices=["gru", "transformer"])
    p.add_argument("--stroke-pooling", action="store_true",
                   help="SCAN-style stroke units; measured worse than point level")
    p.add_argument("--aux-stroke-weight", type=float, default=0.0,
                   help="weight of the per-stroke symbol loss from <traceGroup>")
    p.add_argument("--online-input", default="traj", choices=["traj", "srt"],
                   help="'srt' feeds the token sequence predicted from the "
                        "trajectory, leaving the thesis architecture untouched")
    p.add_argument("--online-dropout", type=float, default=0.0,
                   help="chance of hiding the online stream during training, so "
                        "the decoder stays able to fall back on the image")
    p.add_argument("--srt-dir", default=None,
                   help="folder of predicted SRT files, one per split")
    p.add_argument("--suffix", default="traj3",
                   help="checkpoint folder suffix, one per encoder revision")
    p.add_argument("--max-epochs", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Fix the seed so every ablation variant trains under identical conditions (fair comparison)
    seed_everything(7)

    FUSION = args.fusion
    BIDIRECTIONAL = not args.unidirectional
    # Each variant is saved to its OWN folder so checkpoints never overwrite another model's.
    run_name = FUSION + ("" if BIDIRECTIONAL else "_uni")
    # One folder per online-encoder revision so architectures never share
    # checkpoints: _traj = original, _traj2 = deep conv front end + feature
    # standardisation + distortion, _traj3 = the same plus a TAP-style BiGRU
    # in place of the Transformer encoder, _aux = _traj3 plus the per-stroke
    # symbol loss.
    out_dir = f"lightning_logs/abl_{run_name}_{args.suffix}"

    model = LitBTTR(d_model=256,
    growth_rate=24,
    num_layers=16,
    nhead= 8,
    dim_feedforward= 1024,
    dropout= 0.3,
    num_encoder_layers = 3,
    num_decoder_layers= 3,
    beam_size= 10,
    max_len= 200,
    alpha= 1.0,
    learning_rate= 1.0,
    patience= 20,
    fusion= FUSION,
    bidirectional= BIDIRECTIONAL,
    traj_encoder= args.traj_encoder,   # TAP-style [33] recurrent encoder
    # stroke-level pooling [22] measured worse than point level (token 43.08 vs
    # 47.69 on 2014), so the fusion runs use the point-level BiGRU
    traj_stroke_pooling= args.stroke_pooling,
    # per-stroke symbol supervision from <traceGroup>, training only
    aux_stroke_weight= args.aux_stroke_weight,
    online_input= args.online_input,
    online_dropout= args.online_dropout,
    )
    # .load_from_checkpoint(r"lightning_logs\crohme\lightning_logs\version_14\checkpoints\epoch=19-step=22800-val_ExpRate=0.4355.ckpt")

    dm = CROHMEDatamodule(batch_size=32, num_workers=5,
                          online_input=args.online_input, srt_dir=args.srt_dir)


    trainer = Trainer(
        default_root_dir=out_dir,
        enable_checkpointing=True,
        callbacks = [
            # patience=15 val checks (= 30 epochs). The default of 3 cut the
            # trajectory runs off at epoch 7, while val_loss was still on the
            # plateau that precedes the recognition signal.
            EarlyStopping(monitor="val_loss", mode="min", patience=15),
            LearningRateMonitor(logging_interval='epoch'), 
            ModelCheckpoint(            
                save_top_k=10,
                monitor= 'val_loss',
                mode='min',
                filename='{epoch}-{step}-{val_loss:.4f}',
                save_weights_only=True,
            )
        ], 
        check_val_every_n_epoch=2,
        max_epochs=args.max_epochs,
        gpus=1,
        fast_dev_run=False,
    )
    print(f"[ablation] variant = {run_name}  ->  saving checkpoints under {out_dir}")
    trainer.fit(model, dm)

