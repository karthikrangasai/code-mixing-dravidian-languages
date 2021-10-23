import torch
import pytorch_lightning as pl


class Freeze(pl.callbacks.BaseFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__()

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze(pl_module.backbone, train_bn=True)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ) -> None:
        pass


FINE_TUNING_MAPPING = {
    "freeze": Freeze,
}
