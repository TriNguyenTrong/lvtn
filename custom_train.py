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

if __name__ == "__main__":
    # Fix the seed so every ablation variant trains under identical conditions (fair comparison)
    seed_everything(7)

    # --- Ablation switches (Section 4): change these per run ---
    FUSION = "dual_shared"        # dual_shared (Ours) | offline | online | concat | cascaded
    BIDIRECTIONAL = True       # True (Ours, L2R+R2L) | False (L2R only)
    # Each variant is saved to its OWN folder so checkpoints never overwrite another model's.
    run_name = FUSION + ("" if BIDIRECTIONAL else "_uni")
    out_dir = f"lightning_logs/abl_{run_name}"

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
    )
    # .load_from_checkpoint(r"lightning_logs\crohme\lightning_logs\version_14\checkpoints\epoch=19-step=22800-val_ExpRate=0.4355.ckpt")

    dm = CROHMEDatamodule(batch_size=32, num_workers=5)


    trainer = Trainer(
        default_root_dir=out_dir,
        enable_checkpointing=True,
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min"),
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
        max_epochs=50,
        gpus=1, 
        fast_dev_run=False,
    )
    print(f"[ablation] variant = {run_name}  ->  saving checkpoints under {out_dir}")
    trainer.fit(model, dm)

