import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    parser = ArgumentParser()
    parser.add_argument("--hpc1", action="store_true")
    parser.add_argument("--hpc2", action="store_true")
    parser.add_argument("--hpc3", action="store_true")
    args = parser.parse_args()

    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()

    wandb_logger = WandbLogger(
        project="Code_Mixing_Sentiment_Classifier",
        group="tests",
        job_type="issue_template_tests",
        name="wandb_testing",
        log_model=True,
        id="",
    )

    if args.hpc1:
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            gpus=2,
            num_nodes=1,
            num_sanity_val_steps=0,
            max_epochs=1,
            accelerator="ddp",
            logger=wandb_logger,
        )
    elif args.hpc2:
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            gpus=1,
            num_nodes=2,
            accelerator="ddp",
            num_sanity_val_steps=0,
            max_epochs=1,
            logger=wandb_logger,
        )
    elif args.hpc3:
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            gpus=2,
            num_nodes=2,
            accelerator="ddp",
            num_sanity_val_steps=0,
            max_epochs=1,
            logger=wandb_logger,
        )

    print(
        trainer.global_rank,
        trainer.training_type_plugin.cluster_environment.master_address(),
        trainer.training_type_plugin.cluster_environment.master_port(),
    )

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    run()
