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
from code_mixing_dravidian_languages.utils import (
    early_stopping,
    checkpoint_callback,
    get_model_and_datamodule,
    lr_monitor,
    get_logger,
)


def main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--operation_type", default="train", type=str)

    parser.add_argument("--finetuning_strategy", default=None, type=str, required=False)

    parser.add_argument(
        "--backbone", default="ai4bharat/indic-bert", type=str, required=False
    )
    parser.add_argument("--language", default="tamil", type=str, required=False)

    parser.add_argument("--max_length", default=256, type=int)

    parser.add_argument("--debug", default=False, type=bool, required=False)

    args = parser.parse_args()

    classifier_configuration = CodeMixingSentimentClassifierConfiguration(
        backbone=args.backbone,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        operation_type=args.operation_type,
    )

    logger_configuration = WANDBLoggerConfiguration(
        group=f"{classifier_configuration.backbone}",
        job_type=f"{classifier_configuration.operation_type}",
        name=f"{classifier_configuration.language}_{classifier_configuration.learning_rate}",
        config=asdict(classifier_configuration),
    )

    model, datamodule = get_model_and_datamodule(classifier_configuration)
    wandb_logger = get_logger(logger_configuration)
    callbacks = [early_stopping, checkpoint_callback, lr_monitor]

    if classifier_configuration.finetuning_strategy is not None:
        finetuning_fn = FINE_TUNING_MAPPING[
            classifier_configuration.finetuning_strategy
        ]
        callbacks.append(finetuning_fn())

    trainer = Trainer(
        fast_dev_run=args.debug,
        gpus=1,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        weights_summary="top",
        max_epochs=classifier_configuration.max_epochs,
    )

    wandb_logger.watch(model)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()
