from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

test_year = "2014"
ckp_path = r"lightning_logs/crohme_onoff/lightning_logs/version_1/checkpoints/epoch=49-step=56950-val_loss=0.1534.ckpt"

if __name__ == "__main__":
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year)

    model = LitBTTR.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)
