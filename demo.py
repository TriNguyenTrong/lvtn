import torch

# from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR
print(torch.cuda.device_count())
# print('-------------------asdsad----')
print(torch.cuda.is_available())
