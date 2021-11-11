from argparse import ArgumentParser
from typing import Optional, Union
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from code_mixing_dravidian_languages import DATA_FOLDER_PATH

from code_mixing_dravidian_languages.src import (
    CodeMixingSentimentClassifierDataModule,
    CodeMixingSentimentClassifier,
)
from code_mixing_dravidian_languages.src.data import DATA_METADATA
from code_mixing_dravidian_languages.src.finetuning import FINE_TUNING_MAPPING


def main(
    backbone: str,
    learning_rate: float,
    lr_scheduler: str,
    num_warmup_steps:Union[int, float],
    dataset: str,
    language: str,
    batch_size: int,
    max_length: int,
    num_workers: int,
    operation_type: str,
    finetuning_strategy: str,
    max_epochs: int, 
    gpus: int,
    ckpt_path: str,
    wandb_run_id: str,
    disable_wandb: bool,
    debug: bool,
) -> None:

    # Setup Data
    datamodule = CodeMixingSentimentClassifierDataModule(
        backbone=backbone,
        language=language,
        batch_size=batch_size,
        max_length=max_length,
        padding="max_length",
        num_workers=num_workers,
        dataset=dataset,
    )

    # Setup Model
    if ckpt_path == "":
        model = CodeMixingSentimentClassifier(
            backbone=backbone,
            num_classes=DATA_METADATA[dataset]["num_classes"],
            learning_rate=learning_rate,
            batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        model = CodeMixingSentimentClassifier.load_from_checkpoint(checkpoint_path=ckpt_path)

    if not disable_wandb:
        wandb_logger = WandbLogger(
            project="Code_Mixing_Sentiment_Classifier",
            name=f"{backbone}_{operation_type}_{language}_{learning_rate}",
            log_model=True,
            id=wandb_run_id,
            config={
                "backbone": backbone,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_length": max_length,
                "max_epochs": max_epochs,
                "operation_type": operation_type,
                "num_workers": num_workers,
            },
        )
    else:
        wandb_logger = True

    callbacks = [
        # EarlyStopping Callback
        EarlyStopping(
            monitor="val_accuracy_epoch",
            patience=3,
            mode="max",
        ),
        # Checkpoint Callback
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
            monitor="val_accuracy_epoch",
            save_top_k=1,
            mode="max",
        ),
        # LR Monitor
        LearningRateMonitor(logging_interval="step"),
    ]
    if finetuning_strategy is not None:
        finetuning_fn = FINE_TUNING_MAPPING[finetuning_strategy]
        callbacks.append(finetuning_fn())

    if gpus == 1:
        hardware_settings = {"gpus": 1}
    elif gpus == 12:
        hardware_settings = {"num_nodes": 1, "gpus": 2, "accelerator": "ddp"}
    elif gpus == 21:
        hardware_settings = {"num_nodes": 2, "gpus": 1, "accelerator": "ddp"}
    elif gpus == 22:
        hardware_settings = {"num_nodes": 2, "gpus": 2, "accelerator": "ddp"}
    else:
        hardware_settings = {"gpus": 0}

    trainer = Trainer(
        fast_dev_run=debug,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        weights_summary="top",
        max_epochs=max_epochs,
        accumulate_grad_batches=2,
        num_sanity_val_steps=0,
        **hardware_settings,
    )

    if not disable_wandb:
        wandb_logger.watch(model)

    trainer.fit(model, datamodule=datamodule)

    if not disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    pl.seed_everything(42)
    parser = ArgumentParser()

    parser.add_argument("--backbone", default="ai4bharat/indic-bert", type=str, required=False)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--lr_scheduler", type=str, default="linear", required=False)
    parser.add_argument("--num_warmup_steps", type=Union[int, float], default=0.1, required=False)
    parser.add_argument("--dataset", required=True, type=str, choices=["fire_2020", "codalab"])
    parser.add_argument("--language", required=True, type=str, choices=["all", "tamil", "malayalam", "kannada"])
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--num_workers", type=int, required=False, default=0)
    parser.add_argument("--operation_type", default="train", type=str)
    parser.add_argument("--finetuning_strategy", default=None, type=str, required=False)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--gpus", choices=[0, 1, 12, 21, 22], default=1, type=int, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False, default="")
    parser.add_argument("--wandb_run_id", type=str, required=False, default=None)
    parser.add_argument("--disable_wandb", action="store_true", required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    args = parser.parse_args()

    if (args.ckpt_path == "" and args.wandb_run_id is not None) or (args.ckpt_path != "" and args.wandb_run_id is None):
        print("If loading from checkpoint, provide a wandb run id as well.")

    main(
        backbone=args.backbone,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        num_warmup_steps=args.num_warmup_steps,
        
        dataset=args.dataset,
        language=args.language,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        
        operation_type=args.operation_type,
        finetuning_strategy=args.finetuning_strategy,
        
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        ckpt_path=args.ckpt_path,
        wandb_run_id=args.wandb_run_id,
        
        disable_wandb=args.disable_wandb,
        debug=args.debug,
    )
