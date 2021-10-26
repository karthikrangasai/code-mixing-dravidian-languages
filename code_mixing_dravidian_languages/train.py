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

    parser.add_argument(
        "--gpus", choices=[1, 12, 21, 22], default=1, type=int, required=False
    )
    parser.add_argument("--disable_wandb", action="store_true", required=False)
    parser.add_argument("--ckpt_path", type=str, required=False, default="")
    parser.add_argument("--num_workers", type=int, required=False, default=0)
    parser.add_argument("--debug", default=False, type=bool, required=False)

    args = parser.parse_args()

    classifier_configuration = CodeMixingSentimentClassifierConfiguration(
        backbone=args.backbone,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        operation_type=args.operation_type,
        num_workers=args.num_workers,
    )
    model, datamodule = get_model_and_datamodule(classifier_configuration)

    if not args.disable_wandb:
        logger_configuration = WANDBLoggerConfiguration(
            group=f"{classifier_configuration.backbone}",
            job_type=f"{classifier_configuration.operation_type}",
            name=f"{classifier_configuration.language}_{classifier_configuration.learning_rate}",
            config=asdict(classifier_configuration),
        )
        wandb_logger = get_logger(logger_configuration)
    else:
        wandb_logger = True

    callbacks = [early_stopping, checkpoint_callback, lr_monitor]
    if classifier_configuration.finetuning_strategy is not None:
        finetuning_fn = FINE_TUNING_MAPPING[
            classifier_configuration.finetuning_strategy
        ]
        callbacks.append(finetuning_fn())

    if args.gpus == 1:
        hardware_settings = {"gpus": 1}
    elif args.gpus == 12:
        hardware_settings = {"num_nodes": 1, "gpus": 2, "accelerator": "ddp"}
    elif args.gpus == 21:
        hardware_settings = {"num_nodes": 2, "gpus": 1, "accelerator": "ddp"}
    elif args.gpus == 22:
        hardware_settings = {"num_nodes": 2, "gpus": 2, "accelerator": "ddp"}
    else:
        hardware_settings = {"gpus": 0}

    trainer = Trainer(
        fast_dev_run=args.debug,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        weights_summary="top",
        max_epochs=classifier_configuration.max_epochs,
        **hardware_settings,
    )

    if not args.disable_wandb:
        wandb_logger.watch(model)

    if args.ckpt_path != "":
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
