import os
from argparse import ArgumentParser
from functools import partial

import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from code_mixing_dravidian_languages import DATA_FOLDER_PATH

from code_mixing_dravidian_languages.src import (
    CodeMixingSentimentClassifierDataModule,
    CodeMixingSentimentClassifier,
)
from code_mixing_dravidian_languages.src.finetuning import FINE_TUNING_MAPPING


def sweep_iteration(num_epochs: int, num_workers: int, gpus: int, data_folder_path: str):
    with wandb.init() as run:
        config = run.config

        callbacks = []
        if config.finetuning_strategy is not None:
            finetuning_fn = FINE_TUNING_MAPPING[config.finetuning_strategy]
            callbacks.append(finetuning_fn())

        # Setup Data
        datamodule = CodeMixingSentimentClassifierDataModule(
            backbone=config.backbone,
            language="tamil",
            batch_size=config.batch_size,
            max_length=config.max_length,
            padding="max_length",
            num_workers=num_workers,
            data_folder_path=data_folder_path,
        )

        # Setup Model
        model = CodeMixingSentimentClassifier(
            num_classes=5,
            backbone=config.backbone,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
        )

        wandb_logger = WandbLogger(
            project="Code_Mixing_Sentiment_Classifier",
            group=config.backbone,
            job_type="hparams_search",
            name=f"tamil_{config.learning_rate}",
            config={
                "backbone": config.backbone,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "max_epochs": num_epochs,
                "operation_type": "train" if config.finetuning_strategy is None else "finetune",
                "num_workers": num_workers,
            },
        )

        if gpus == 1:
            hardware_settings = {"gpus": 1}
        elif gpus == 12:
            hardware_settings = {
                "num_nodes": 1,
                "gpus": 2,
                "accelerator": "ddp",
            }
        elif gpus == 21:
            hardware_settings = {
                "num_nodes": 2,
                "gpus": 1,
                "accelerator": "ddp",
            }
        elif gpus == 22:
            hardware_settings = {
                "num_nodes": 2,
                "gpus": 2,
                "accelerator": "ddp",
            }
        else:
            hardware_settings = {"gpus": 0}

        trainer = Trainer(
            log_every_n_steps=10,
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=num_epochs,
            accumulate_grad_batches=2,
            num_sanity_val_steps=0,
            **hardware_settings,
        )

        wandb_logger.watch(model)
        try:
            trainer.fit(model, datamodule=datamodule)
        except RuntimeError:
            wandb.log({"val_accuracy_epoch": -1.0})
        wandb.finish()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None, required=False)
    parser.add_argument("--sampler", type=str, default="random", required=False, choices=["random", "grid", "bayes"])
    parser.add_argument("--trials", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--num_workers", type=int, required=False, default=0)
    parser.add_argument("--gpus", choices=[1, 12, 21, 22], default=1, type=int, required=False)
    parser.add_argument("--data_folder_path", required=False, default=DATA_FOLDER_PATH)
    args = parser.parse_args()

    sweep_config = {
        "method": args.sampler,  # Random search
        "metric": {
            # We want to maximize val_accuracy
            "name": "val_accuracy_epoch",
            "goal": "maximize",
        },
        "parameters": {
            "backbone": {"values": ["ai4bharat/indic-bert", "xlm-roberta-base", "xlm-roberta-large"]},
            "batch_size": {"values": [2, 4, 8, 16]},
            "max_length": {"values": [128, 256, 384, 512]},
            "learning_rate": {"values": [1e-7, 5.5e-7, 1e-6, 5.5e-6, 1e-5, 5.5e-5, 1e-4, 5.5e-4, 1e-3]},
            "finetuning_strategy": {"values": [None, "freeze"]},
        },
    }

    wandb.login()
    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project="Code_Mixing_Sentiment_Classifier")
    else:
        sweep_id = args.sweep_id
    wandb.agent(
        sweep_id,
        function=partial(
            sweep_iteration,
            num_epochs=args.epochs,
            num_workers=args.num_workers,
            gpus=args.gpus,
            data_folder_path=args.data_folder_path,
        ),
        project="Code_Mixing_Sentiment_Classifier",
        count=args.trials,
    )
