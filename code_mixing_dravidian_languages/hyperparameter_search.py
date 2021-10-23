import os
import wandb
from argparse import ArgumentParser
from dataclasses import asdict

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from code_mixing_dravidian_languages.src import (
    CodeMixingSentimentClassifierConfiguration,
    WANDBLoggerConfiguration,
)
from code_mixing_dravidian_languages.src.finetuning import FINE_TUNING_MAPPING
from code_mixing_dravidian_languages.utils import get_model_and_datamodule, get_logger


def sweep_iteration():
    with wandb.init() as run:
        config = run.config

        classifier_configuration = CodeMixingSentimentClassifierConfiguration(
            backbone=config.backbone,
            learning_rate=config.lr,
            batch_size=config.batch_size,
            max_length=config.max_length,
            max_epochs=3,
            operation_type="hparams_search",
            finetuning_strategy=config.finetuning_strategy,
        )
        logger_configuration = WANDBLoggerConfiguration(
            group=f"{classifier_configuration.backbone}",
            job_type=f"{classifier_configuration.operation_type}",
            name=f"{classifier_configuration.language}_{classifier_configuration.learning_rate}",
            config=asdict(classifier_configuration),
        )

        callbacks = []
        if classifier_configuration.finetuning_strategy is not None:
            finetuning_fn = FINE_TUNING_MAPPING[
                classifier_configuration.finetuning_strategy
            ]
            callbacks.append(finetuning_fn())

        model, datamodule = get_model_and_datamodule(classifier_configuration)
        wandb_logger = get_logger(logger_configuration)
        trainer = Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            gpus=1,
            num_nodes=2,
            accelerator="ddp",
            max_epochs=3,
        )

        wandb_logger.watch(model)
        try:
            trainer.fit(model, datamodule=datamodule)
        except RuntimeError:
            wandb.log({"val_accuracy_epoch": -1.0})
        wandb.finish()


sweep_config = {
    "method": "random",  # Random search
    "metric": {
        # We want to maximize val_accuracy
        "name": "val_accuracy_epoch",
        "goal": "maximize",
    },
    "parameters": {
        "backbone": {
            # Choose from pre-defined values
            "values": ["ai4bharat/indic-bert", "xlm-roberta-base", "xlm-roberta-large"]
        },
        "batch_size": {
            # Choose from pre-defined values
            "values": [4, 8, 16]
        },
        "max_length": {
            # Choose from pre-defined values
            "values": [128, 258, 384, 512]
        },
        "lr": {
            # log uniform distribution between exp(min) and exp(max)
            "distribution": "log_uniform",
            "min": -18.420680744,  # exp(-18.420680744) = 1e-8
            "max": 0,  # exp(0) = 1
        },
        "finetuning_strategy": {
            # Choose from pre-defined values
            "values": [None, "freeze"]
        },
    },
}


def main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--count", default=1, type=int)
    args = parser.parse_args()

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Code_Mixing_Sentiment_Classifier")
    wandb.agent(
        sweep_id,
        function=sweep_iteration,
        project="Code_Mixing_Sentiment_Classifier",
        count=args.count,
    )


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    main()
