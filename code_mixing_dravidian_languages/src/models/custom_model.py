from typing import Callable, Dict, List, Optional, Union

import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.nn import Mish, ReLU, Tanh
from torch.optim import Adam, AdamW
from torch_optimizer import Ranger

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from datasets import load_metric
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from code_mixing_dravidian_languages.src.focal_loss import focal_loss

ACTIVATIONS = {
    "mish": Mish,
    "relu": ReLU,
    "tanh": Tanh,
}

OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "ranger": Ranger,
}

class CustomClassifier(torch.nn.Module):
    def __init__(
        self,
        backbone: str,
        num_labels: int,
        linear_layers: List[int],
        activation: str,
        classifier_dropout: Optional[float] = None
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(backbone)
        
        if classifier_dropout is None:
            _classifier_dropout = getattr(self.backbone.config, "classifier_dropout", None)
            _classifier_dropout_prob = getattr(self.backbone.config, "classifier_dropout_prob", None)
            if (_classifier_dropout or _classifier_dropout_prob) is not None:
                classifier_dropout = _classifier_dropout or _classifier_dropout_prob
            else:
                classifier_dropout = self.backbone.config.hidden_dropout_prob
        
        hidden_size = self.backbone.config.hidden_size

        linear_layers = [hidden_size] + linear_layers + [num_labels]
        layers = []
        for _in, _out in zip(linear_layers, linear_layers[1:]):
            layers += [torch.nn.Dropout(classifier_dropout), torch.nn.Linear(_in, _out), ACTIVATIONS[activation]()]
        self.head = torch.nn.Sequential(*layers)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = output.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]  # take [CLS] token
        x = self.head(cls_token)
        return x



class CodeMixingCustomSentimentClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "ai4bharat/indic-bert",
        batch_size: int = 8,
        num_classes: int = 5,
        learning_rate: float = 1e-5,
        lr_scheduler: str = "linear",
        num_warmup_steps: Union[int, float] = 0.05,
        weight_decay: float = 0.01,
        gamma: float = 0.1,
        reduction: str = "mean",
        dropout: float = 0.2,
        linear_layers: List[int] = [256, 32],
        loss_fn: str = "focal_loss",
        optimizer: str = "adam",
        activation_fn: str = "relu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.activation_fn = activation_fn

        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay = weight_decay

        self.model = CustomClassifier(
            backbone=backbone,
            num_labels=self.num_classes,
            linear_layers=linear_layers,
            classifier_dropout=dropout,
            activation=self.activation_fn,
        )
        self.gamma = gamma
        self.reduction = reduction
        self.loss_fn = loss_fn

        self.f1_metric = load_metric("f1")
        self.accuracy = load_metric("accuracy")

    def forward(
        self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        input_ids = inputs["input_ids"].view(self.batch_size, -1)
        attention_mask = inputs["attention_mask"].view(self.batch_size, -1)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _common_step(self, batch, batch_idx, prefix: str) -> torch.Tensor:
        target: torch.Tensor = batch.pop("labels")
        inputs: Dict[str, torch.Tensor] = batch
        preds: torch.Tensor = self(inputs, target)

        dataloader: Callable = getattr(self, f"{prefix}_dataloader")
        class_weights = torch.tensor(dataloader().dataset.class_weights, dtype=torch.float, device=self.device)
        if self.loss_fn == "focal_loss":
            loss = focal_loss(preds=preds, target=target, alpha=class_weights, gamma=self.gamma, reduction=self.reduction)
        elif self.loss_fn == "cross_entropy":
            loss = F.cross_entropy(input=preds, target=target, weight=class_weights)

        preds = torch.argmax(torch.log_softmax(preds, 0), 1)
        acc = self.accuracy.compute(predictions=preds, references=target)
        f1_macro = self.f1_metric.compute(predictions=preds, references=target, average="macro")
        f1_micro = self.f1_metric.compute(predictions=preds, references=target, average="macro")
        f1_weighted = self.f1_metric.compute(predictions=preds, references=target, average="weighted")
        metrics = {
            f"{prefix}_accuracy": acc["accuracy"],
            f"{prefix}_f1_macro": f1_macro["f1"],
            f"{prefix}_f1_micro": f1_micro["f1"],
            f"{prefix}_f1_weighted": f1_weighted["f1"],
        }
        
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, prefix="test")

    def get_backbone(self):
        return self.model.backbone

    def get_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if not getattr(self, "trainer", None):
            raise MisconfigurationException("The LightningModule isn't attached to the trainer yet.")
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            num_batches = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.train_dataloader())
            num_batches = int(dataset_size * self.trainer.limit_train_batches)
        else:
            num_batches = len(self.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (num_batches // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    @staticmethod
    def _compute_warmup(num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        if isinstance(num_warmup_steps, float) and (num_warmup_steps > 1 or num_warmup_steps < 0):
            raise MisconfigurationException("`num_warmup_steps` as float should be provided between 0 and 1.")

        if isinstance(num_warmup_steps, int):
            if num_warmup_steps > num_training_steps:
                raise MisconfigurationException("`num_warmup_steps` as int should be less than `num_training_steps`.")
            return num_warmup_steps
            

        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return round(num_warmup_steps)

    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = OPTIMIZERS[self.optimizer](optimizer_grouped_parameters, lr=self.learning_rate)

        if self.lr_scheduler is not None:
            num_training_steps = self.get_num_training_steps()
            lr_scheduler = get_scheduler(
                name=self.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self._compute_warmup(num_training_steps, self.num_warmup_steps),
                num_training_steps=num_training_steps,
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler_config]

        return optimizer
