from itertools import accumulate
from typing import List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning


class TestFineTuning(BaseFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__()

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        pass

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ) -> None:
        print(f"Epoch Number: {epoch}")


class Freeze(BaseFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__()
        self.train_bn=train_bn

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(pl_module.get_backbone(), train_bn=self.train_bn)

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
        self.freeze(pl_module.get_backbone(), train_bn=self.train_bn)

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
            modules=pl_module.get_backbone(),
            optimizer=optimizer,
            train_bn=self.train_bn,
        )

class GradualUnFreezing(BaseFinetuning):
    def __init__(
        self, 
        init_lr: float = 5e-5,
        train_bn: bool = False,
        discriminative_finetuning: Optional[float] = None,
        num_epochs_train_only_head: int = 1,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.curr_layer = 1
        self.train_bn = train_bn
        self.discriminative_finetuning = discriminative_finetuning
        self.num_epochs_train_only_head = num_epochs_train_only_head


    def _get_layer_splits(self, num_layers, num_epochs) -> List[Tuple[int, int]]:
        layer_splits = []
        rem = num_layers % num_epochs
        layer_splits += [(num_layers // num_epochs) + 1] * rem
        layer_splits += [(num_layers // num_epochs)] * (num_epochs-rem)
        layer_splits = list(accumulate(layer_splits))
        layer_splits = [0] + layer_splits
        layer_splits = [(layer_splits[i], layer_splits[i+1]) for i in range(0, len(layer_splits)-1)]
        return layer_splits


    def freeze_before_training(self, pl_module: LightningModule) -> None:
        backbone: torch.nn.Module = pl_module.get_backbone()
        self.freeze(backbone, train_bn=self.train_bn)
        self.layer_names = []
        for n, c in backbone.named_children():
            if n == "encoder":
                for l, _c in c.layer.named_children():
                    self.layer_names.append(f"{n}.layer.{l}.")
            else:
                self.layer_names.append(f"{n}.")
        self.layer_names = self.layer_names[::-1]
        self.num_layers = len(self.layer_names)
        self.layer_splits = self._get_layer_splits(
            self.num_layers, pl_module.trainer.max_epochs - self.num_epochs_train_only_head
        )

    def finetune_function(
        self,
        pl_module: LightningModule,
        current_epoch: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ) -> None:
        # current_epoch is 0-indexed
        if current_epoch + 1 > self.num_epochs_train_only_head:
            # Fetch the backbone
            backbone: torch.nn.Module = pl_module.get_backbone()

            # Fetch the layer names
            indices = self.layer_splits[current_epoch - self.num_epochs_train_only_head]
            layer_names_to_be_unfrozen = self.layer_names[indices[0] : indices[1]]

            # Map each layer name to modules of the layer
            layers_to_be_unfrozen = {layer_name: [] for layer_name in layer_names_to_be_unfrozen}
            for n, module in backbone.named_modules():
                for layer_name in layers_to_be_unfrozen:
                    if n.startswith(layer_name):
                        layers_to_be_unfrozen[layer_name].append(module)

            if len(layers_to_be_unfrozen) > 0:
                if self.discriminative_finetuning is not None:
                    for layer_name, modules in layers_to_be_unfrozen.items():
                        if len(modules) > 0:
                            self.unfreeze_and_add_param_group(
                                modules=modules,
                                optimizer=optimizer,
                                train_bn=self.train_bn,
                                lr=self.init_lr / pow(self.discriminative_finetuning, self.curr_layer),
                            )
                            self.curr_layer += 1


def parse_finetuning_arguments(strategy: str, value: str) -> Tuple[str, Union[int, float]]:
    if strategy in ["freeze", "test"]:
        value = None
    elif strategy == "freeze_unfreeze":
        value = int(value)
    elif strategy == "gradual_unfreezing":
        value = float(value)
    return (strategy, value)


FINE_TUNING_MAPPING = {
    "test": TestFineTuning,
    "freeze": Freeze,
    "freeze_unfreeze": FreezeUnfreeze,
    "gradual_unfreezing": GradualUnFreezing,
}
