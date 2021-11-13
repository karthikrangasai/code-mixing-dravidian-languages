import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning


class Freeze(BaseFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__()

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(pl_module.backbone, train_bn=True)

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ) -> None:
        pass

class FreezeUnfreeze(BaseFinetuning):
    def __init__(self, train_bn: bool = True, unfreeze_epoch: int = 10):
        super().__init__()
        self.train_bn = train_bn
        self.unfreeze_epoch = unfreeze_epoch

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(pl_module.backbone, train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch != self.unfreeze_epoch:
            return
        self.unfreeze_and_add_param_group(
            modules=pl_module.backbone,
            optimizer=optimizer,
            train_bn=self.train_bn,
        )


FINE_TUNING_MAPPING = {
    "freeze": Freeze,
    "freeze_unfreeze": FreezeUnfreeze,
}
