from dataclasses import asdict
from typing import Tuple

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger

from code_mixing_dravidian_languages.src import (
    CodeMixingSentimentClassifierConfiguration,
    WANDBLoggerConfiguration,
)
from code_mixing_dravidian_languages.src import CodeMixingSentimentClassifierDataModule
from code_mixing_dravidian_languages.src import CodeMixingSentimentClassifier


# EarlyStopping Callback
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    mode="max",
)

# Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
    monitor="val_accuracy",
    save_top_k=1,
    mode="max",
)


# LR Monitor
lr_monitor = LearningRateMonitor(logging_interval="step")

# WANDB Logger
def get_logger(configuration: WANDBLoggerConfiguration) -> WandbLogger:
    return WandbLogger(
        project="Code_Mixing_Sentiment_Classifier",
        log_model=True,
        **asdict(configuration),
    )


# Create Model and Datamodule
def get_model_and_datamodule(
    configuration: CodeMixingSentimentClassifierConfiguration,
) -> Tuple[CodeMixingSentimentClassifier, CodeMixingSentimentClassifierDataModule]:

    # setup data
    datamodule = CodeMixingSentimentClassifierDataModule(
        backbone=configuration.backbone,
        language=configuration.language,
        batch_size=configuration.batch_size,
        max_length=configuration.max_length,
        padding="max_length",
    )

    # setup model - note how we refer to sweep parameters with wandb.config
    model = CodeMixingSentimentClassifier(
        backbone=configuration.backbone,
        num_classes=5,
        learning_rate=configuration.learning_rate,
        batch_size=configuration.batch_size,
    )

    return model, datamodule
