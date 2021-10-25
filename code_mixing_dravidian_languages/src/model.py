from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from datasets import load_metric
from transformers import AutoModel
from transformers.optimization import AdamW


class CodeMixingSentimentClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        batch_size: int,
        learning_rate: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AutoModel.from_pretrained(backbone)
        self.num_classes = num_classes
        hidden_size = self.backbone.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, self.num_classes)

        self.learning_rate = learning_rate

        self.batch_size = batch_size

        self.f1_metric = load_metric("f1")
        self.accuracy = load_metric("accuracy")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].view(self.batch_size, -1)
        attention_mask = batch["attention_mask"].view(self.batch_size, -1)
        backbone_output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        preds = self.classifier(backbone_output[0][:, 0])
        return preds

    def _common_step(self, batch, batch_idx, prefix: str) -> torch.Tensor:
        y: torch.Tensor = batch.pop("labels")
        x: Dict[str, torch.Tensor] = batch
        y_hat: torch.Tensor = self(x)
        preds = torch.argmax(torch.log_softmax(y_hat, 0), 1)

        y = y.view(-1)
        y_hat = y_hat.view(-1, self.num_classes)

        loss = F.cross_entropy(y_hat, y)

        acc = self.accuracy.compute(predictions=preds, references=y)
        f1_macro = self.f1_metric.compute(
            predictions=preds, references=y, average="macro"
        )
        f1_micro = self.f1_metric.compute(
            predictions=preds, references=y, average="macro"
        )
        f1_weighted = self.f1_metric.compute(
            predictions=preds, references=y, average="weighted"
        )

        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        metrics = {
            f"{prefix}_accuracy": acc["accuracy"],
            f"{prefix}_f1_macro": f1_macro["f1"],
            f"{prefix}_f1_micro": f1_micro["f1"],
            f"{prefix}_f1_weighted": f1_weighted["f1"],
        }
        self.log_dict(metrics, logger=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        _model_parameters = []

        for p in self.backbone.parameters():
            if p.requires_grad:
                _model_parameters.append(p)

        for p in self.classifier.parameters():
            if p.requires_grad:
                _model_parameters.append(p)

        optimizer = AdamW(
            _model_parameters,
            lr=self.learning_rate,
        )

        scheduler = None

        if scheduler is None:
            return optimizer

        return [optimizer], [scheduler]
