from typing import Dict, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import functional as F
from datasets import load_metric
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import AdamW, get_scheduler

class CodeMixingHFSentimentClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "ai4bharat/indic-bert",
        batch_size: int = 8,
        num_classes: int = 5,
        learning_rate: float = 1e-5,
        lr_scheduler: str = "linear",
        num_warmup_steps: Union[int, float] = 0.05,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model: torch.nn.Module = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=self.num_classes
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay = weight_decay
        self.f1_metric = load_metric("f1")
        self.accuracy = load_metric("accuracy")

    def forward(self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> SequenceClassifierOutput:
        input_ids = inputs["input_ids"].view(self.batch_size, -1)
        attention_mask = inputs["attention_mask"].view(self.batch_size, -1)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _common_step(self, batch, batch_idx, prefix: str) -> torch.Tensor:
        y: torch.Tensor = batch.pop("labels")
        x: Dict[str, torch.Tensor] = batch
        y_hat: SequenceClassifierOutput = self(x, y)

        loss: torch.FloatTensor = y_hat.loss
        logits: torch.FloatTensor = y_hat.logits

        preds = torch.argmax(torch.log_softmax(logits, 0), 1)
        acc = self.accuracy.compute(predictions=preds, references=y)
        f1_macro = self.f1_metric.compute(predictions=preds, references=y, average="macro")
        f1_micro = self.f1_metric.compute(predictions=preds, references=y, average="macro")
        f1_weighted = self.f1_metric.compute(predictions=preds, references=y, average="weighted")
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
        model_type = self.model.config.model_type
        if model_type == "albert":
            return self.model.albert
        elif model_type == "bert":
            return self.model.bert
        elif model_type == "xlm-roberta":
            return self.model.roberta
        elif model_type == "distilbert":
            return self.model.distilbert

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, correct_bias=True)

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
