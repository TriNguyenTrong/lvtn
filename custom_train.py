from pytorch_lightning import Trainer
from bttr.datamodule import CROHMEDatamodule, vocab
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
    model = LitBTTR(d_model=256,
    growth_rate=24,
    num_layers=16,
    nhead= 8,
    dim_feedforward= 1024,
    dropout= 0.3,
    num_decoder_layers= 3,
    beam_size= 10,
    max_len= 200,
    alpha= 1.0,
    learning_rate= 1.0,
    patience= 20,
    )
    # .load_from_checkpoint(r"lightning_logs\crohme\lightning_logs\version_14\checkpoints\epoch=19-step=22800-val_ExpRate=0.4355.ckpt")

    dm = CROHMEDatamodule(batch_size=32, num_workers=5)


    trainer = Trainer(
        default_root_dir='lightning_logs/crohme',
        enable_checkpointing=True,
        callbacks = [
            EarlyStopping(monitor="val_ExpRate", mode="max"),
            LearningRateMonitor(logging_interval='epoch'), 
            ModelCheckpoint(            
                save_top_k=1,
                monitor= 'val_ExpRate',
                mode='max',
                filename='{epoch}-{step}-{val_ExpRate:.4f}',
                save_weights_only=True,
            )
        ], 
        check_val_every_n_epoch=2,
        max_epochs=50,
        gpus=1, 
        fast_dev_run=True,
    )
    trainer.fit(model, dm)

