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
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--val-split", default=None,
                   help="file of training-sample names held out as validation; "
                        "without it validation runs on CROHME 2014 as before")
    p.add_argument("--lr-schedule", default="legacy", choices=["legacy", "plateau"],
                   help="legacy = the mode='max' step decay of every thesis number")
    p.add_argument("--monitor", default="val_loss", choices=["val_loss", "val_ExpRate"],
                   help="metric for plateau / early stopping / checkpoint selection "
                        "(val_ExpRate needs --val-exprate)")
    p.add_argument("--val-exprate", action="store_true",
                   help="beam-search the validation set to log val_ExpRate")
    p.add_argument("--lr-patience", type=int, default=10,
                   help="plateau: validation checks without improvement before lr x0.1")
    p.add_argument("--es-patience", type=int, default=15,
                   help="early stopping patience, in validation checks")
    p.add_argument("--sa-residual", action="store_true",
                   help="decoder: residual stream carries the self-attention output x_q")
    p.add_argument("--separate-cross", action="store_true",
                   help="decoder: own attention + norm for the second cross-attention")
    p.add_argument("--cascaded-residual", default="xq", choices=["xq", "x"])
    p.add_argument("--out-dir", default=None,
                   help="checkpoint root; default lightning_logs/abl_<fusion>_<suffix>")
    p.add_argument("--gpu-mem-fraction", type=float, default=0.0,
                   help="cap this process at a share of GPU memory (0 = no cap); for running "
                        "two trainings side by side")
    p.add_argument("--check-val-every-n-epoch", type=int, default=2)
    p.add_argument("--limit-train-batches", type=float, default=1.0,
                   help="smoke tests only; a value above 1 counts batches")
    p.add_argument("--limit-val-batches", type=float, default=1.0,
                   help="smoke tests only; a value above 1 counts batches")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Fix the seed so every ablation variant trains under identical conditions (fair comparison)
    seed_everything(args.seed)

    if args.gpu_mem_fraction > 0:
        import torch
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    FUSION = args.fusion
    BIDIRECTIONAL = not args.unidirectional
    # Each variant is saved to its OWN folder so checkpoints never overwrite another model's.
    run_name = FUSION + ("" if BIDIRECTIONAL else "_uni")
    # One folder per online-encoder revision so architectures never share
    # checkpoints: _traj = original, _traj2 = deep conv front end + feature
    # standardisation + distortion, _traj3 = the same plus a TAP-style BiGRU
    # in place of the Transformer encoder, _aux = _traj3 plus the per-stroke
    # symbol loss.
    out_dir = args.out_dir or f"lightning_logs/abl_{run_name}_{args.suffix}"

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
    sa_residual= args.sa_residual,
    separate_cross= args.separate_cross,
    cascaded_residual= args.cascaded_residual,
    lr_schedule= args.lr_schedule,
    lr_patience= args.lr_patience,
    monitor= args.monitor,
    val_exprate= args.val_exprate,
    )
    # .load_from_checkpoint(r"lightning_logs\crohme\lightning_logs\version_14\checkpoints\epoch=19-step=22800-val_ExpRate=0.4355.ckpt")

    dm = CROHMEDatamodule(batch_size=32, num_workers=5,
                          online_input=args.online_input, srt_dir=args.srt_dir,
                          val_split=args.val_split)

    monitor_mode = "max" if args.monitor == "val_ExpRate" else "min"


    trainer = Trainer(
        default_root_dir=out_dir,
        enable_checkpointing=True,
        callbacks = [
            # patience=15 val checks (= 30 epochs). The default of 3 cut the
            # trajectory runs off at epoch 7, while val_loss was still on the
            # plateau that precedes the recognition signal.
            EarlyStopping(monitor=args.monitor, mode=monitor_mode, patience=args.es_patience),
            LearningRateMonitor(logging_interval='epoch'), 
            ModelCheckpoint(            
                save_top_k=10,
                monitor= args.monitor,
                mode=monitor_mode,
                filename='{epoch}-{step}-{' + args.monitor + ':.4f}',
                save_weights_only=True,
            )
        ], 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,
        gpus=1,
        limit_train_batches=int(args.limit_train_batches) if args.limit_train_batches > 1 else args.limit_train_batches,
        limit_val_batches=int(args.limit_val_batches) if args.limit_val_batches > 1 else args.limit_val_batches,
        fast_dev_run=False,
    )
    print(f"[ablation] variant = {run_name}  ->  saving checkpoints under {out_dir}")
    trainer.fit(model, dm)
    if args.gpu_mem_fraction > 0:
        import torch
        print(f"[gpu] peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
              f"peak reserved {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")

