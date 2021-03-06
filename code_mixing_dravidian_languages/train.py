import os
from argparse import ArgumentParser, ArgumentError
from typing import List, Optional, Tuple, Union
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
    MODEL_MAPPING
)
from code_mixing_dravidian_languages.src.data import DATA_METADATA
from code_mixing_dravidian_languages.src.finetuning import FINE_TUNING_MAPPING, parse_finetuning_arguments
from code_mixing_dravidian_languages.src.models.custom_model import ACTIVATIONS, OPTIMIZERS


def main(
    model_type: str,
    backbone: str,
    linear_layers: List[int],
    learning_rate: float,
    optimizer: str,
    activation_fn: str,
    lr_scheduler: str,
    num_warmup_steps:Union[int, float],
    gamma: float,
    reduction: str,
    dropout: float,
    loss_fn: str,
    dataset: str,
    language: str,
    preprocess_fn: str,
    batch_size: int,
    max_length: int,
    num_workers: int,
    operation_type: str,
    finetuning_strategy: Optional[Tuple[str, Union[int, float]]],
    train_bn: bool,
    max_epochs: int, 
    gpus: int,
    accumulate_grad_batches: int,
    deterministic: bool,
    save_dir: str,
    ckpt_path: str,
    wandb_run_id: str,
    early_stopping: bool,
    disable_wandb: bool,
    debug: bool,
) -> None:

    # Setup Data
    datamodule = CodeMixingSentimentClassifierDataModule(
        backbone=backbone,
        language=language,
        preprocess_fn=preprocess_fn,
        batch_size=batch_size,
        max_length=max_length,
        padding="max_length",
        num_workers=num_workers,
        dataset=dataset,
    )

    # Setup Model
    if ckpt_path == "":
        model_settings = {}
        if model_type == "custom":
            model_settings["gamma"] = gamma
            model_settings["reduction"] = reduction
            model_settings["dropout"] = dropout
            model_settings["linear_layers"] = linear_layers
            model_settings["loss_fn"] = loss_fn
            model_settings["optimizer"] = optimizer
            model_settings["activation_fn"] = activation_fn

        model = MODEL_MAPPING[model_type](
            backbone=backbone,
            num_classes=DATA_METADATA[dataset]["num_classes"],
            learning_rate=learning_rate,
            batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            num_warmup_steps=num_warmup_steps,
            **model_settings,
        )
    else:
        model = MODEL_MAPPING[model_type].load_from_checkpoint(checkpoint_path=ckpt_path)

    if not disable_wandb:
        wandb_logger = WandbLogger(
            project="Code_Mixing_Sentiment_Classifier",
            group="focal_loss",
            save_dir=os.path.join(save_dir, "wandb") if save_dir is not None else None,
            name=f"{dataset}_{preprocess_fn}_{backbone}_{operation_type}_{language}_{learning_rate}_{loss_fn}_{finetuning_strategy}",
            log_model=True,
            id=wandb_run_id,
            config={
                "model_type": model_type,
                "backbone": backbone,
                "linear_layers": linear_layers,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "activation_fn": activation_fn,
                "lr_scheduler": lr_scheduler,
                "num_warmup_steps": num_warmup_steps,
                "gamma": gamma,
                "reduction": reduction,
                "dropout": dropout,
                "loss_fn": loss_fn,
                "dataset": dataset,
                "language": language,
                "preprocess_fn": preprocess_fn,
                "batch_size": batch_size,
                "max_length": max_length,
                "num_workers": num_workers,
                "operation_type": operation_type,
                "finetuning_strategy": finetuning_strategy,
                "train_bn": train_bn,
                "max_epochs": max_epochs,
                "gpus": gpus,
                "accumulate_grad_batches": accumulate_grad_batches,
                "deterministic": deterministic,
            },
        )
    else:
        wandb_logger = True

    callbacks = [
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
    
    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_accuracy_epoch", patience=3, mode="max"))
    
    if finetuning_strategy is not None:
        finetuning_fn = FINE_TUNING_MAPPING[finetuning_strategy[0]]
            
        if finetuning_strategy[0] == "freeze_unfreeze":
            finetuning_strategy_metadata = {
                "unfreeze_epoch": finetuning_strategy[1],
                "train_bn": train_bn,
            }
        elif finetuning_strategy[0] == "gradual_unfreezing":
            finetuning_strategy_metadata = {
                "init_lr": learning_rate,
                "train_bn": train_bn,
                "discriminative_finetuning": finetuning_strategy[1],
                "num_epochs_train_only_head": 1,
            }
        else:
            finetuning_strategy_metadata = {"train_bn": train_bn}
            
        callbacks.append(finetuning_fn(**finetuning_strategy_metadata))

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
        deterministic=deterministic,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        weights_summary="top",
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        num_sanity_val_steps=0,
        default_root_dir=save_dir if save_dir is not None else os.getcwd(),
        **hardware_settings,
    )

    if not disable_wandb:
        wandb_logger.watch(model)

    trainer.fit(model, datamodule=datamodule)

    if not disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, required=False)
    
    parser.add_argument("--model_type", default="hf", type=str, choices= ["hf", "custom"], required=False)
    parser.add_argument("--backbone", default="ai4bharat/indic-bert", type=str, required=False)
    parser.add_argument("--linear_layers", nargs='+', type=int, required=False, default=[256, 32])
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--optimizer", type=str, default="adam", choices=OPTIMIZERS.keys())
    parser.add_argument("--activation_fn", type=str, default="relu", choices=ACTIVATIONS.keys())
    parser.add_argument("--lr_scheduler", type=str, default=None, required=False)
    parser.add_argument("--num_warmup_steps", type=float, default=0.1, required=False)
    parser.add_argument("--loss_fn", default="focal_loss", required=False, type=str, choices=["focal_loss", "cross_entropy"])
    parser.add_argument("--gamma", default=0.1, required=False, type=float)
    parser.add_argument("--reduction", default="mean", required=False, type=str)
    parser.add_argument("--dropout", default=0.2, required=False, type=float)
    
    parser.add_argument("--dataset", required=True, type=str, choices=["fire_2020", "fire_2020_trans", "codalab"])
    parser.add_argument("--language", required=True, type=str, choices=["all", "tamil", "malayalam", "kannada"])
    parser.add_argument("--preprocess_fn", required=False, type=str, default=None, choices=[None, "indic", "google"])
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--num_workers", type=int, required=False, default=0)
    
    parser.add_argument("--operation_type", default="train", type=str)
    parser.add_argument("--finetuning_strategy", default=None, type=str, required=False, nargs=2)
    parser.add_argument("--train_bn", action="store_true", required=False)
    
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--gpus", choices=[0, 1, 12, 21, 22], default=1, type=int, required=False)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, required=False)
    parser.add_argument("--deterministic", action="store_true", required=False)
    
    parser.add_argument("--save_dir", default=None, type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False, default="")
    parser.add_argument("--wandb_run_id", type=str, required=False, default=None)
    parser.add_argument("--early_stopping", action="store_true", required=False)
    parser.add_argument("--disable_wandb", action="store_true", required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    args = parser.parse_args()

    if (args.ckpt_path == "" and args.wandb_run_id is not None) or (args.ckpt_path != "" and args.wandb_run_id is None):
        print("If loading from checkpoint, provide a wandb run id as well.")

    if not isinstance(args.finetuning_strategy[0], str):
        raise ArgumentError(message=f"finetuning strategy should be among {FINE_TUNING_MAPPING.keys()}")
    
    args.finetuning_strategy = parse_finetuning_arguments(args.finetuning_strategy[0], args.finetuning_strategy[1])


    pl.seed_everything(args.seed)
    main(
        model_type=args.model_type,
        backbone=args.backbone,
        linear_layers=args.linear_layers,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        activation_fn=args.activation_fn,
        lr_scheduler=args.lr_scheduler,
        num_warmup_steps=args.num_warmup_steps,
        gamma=args.gamma,
        reduction=args.reduction,
        dropout=args.dropout,
        loss_fn=args.loss_fn,
        
        dataset=args.dataset,
        language=args.language,
        preprocess_fn=args.preprocess_fn,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        
        operation_type=args.operation_type,
        finetuning_strategy=tuple(args.finetuning_strategy),
        train_bn=args.train_bn,
        
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=args.deterministic,
        save_dir=args.save_dir,
        ckpt_path=args.ckpt_path,
        wandb_run_id=args.wandb_run_id,
        
        early_stopping=args.early_stopping,
        disable_wandb=args.disable_wandb,
        debug=args.debug,
    )
