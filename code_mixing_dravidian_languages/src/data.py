import os
import pandas as pd
from pathlib import Path
from typing import Dict
from functools import partial

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, default_data_collator

from code_mixing_dravidian_languages import DATA_FOLDER_PATH
from code_mixing_dravidian_languages.src.preprocess import (
    _text_preprocess_fn,
    _category_preprocess_fn,
)


class CodeMixingSentimentClassifierDataset(Dataset):
    def __init__(
        self, filepath: str, tokenizer: str, language: str, max_token_len: int = 128
    ):
        self.data = pd.read_csv(filepath, sep="\t")
        self.tokenizer = tokenizer
        self.language = language
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        preprocessed_text = _text_preprocess_fn(data_row["text"], self.language)
        label = _category_preprocess_fn(data_row["category"])

        sample = self.tokenizer.encode_plus(
            preprocessed_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        sample["label"] = label

        return sample


class CodeMixingSentimentClassifierDataModule(LightningDataModule):
    """

    Args:
        backbone: Hugging Face backbone model to be used.
        DATA_FOLDER_PATH: Folder that has the initial data
        train_val_split: Ratio to split the data files into train and validation sets.
    """

    datasets: Dict[str, Dataset]

    def __init__(
        self,
        backbone: str,
        language: str,
        batch_size: int = 8,
        max_length: int = 128,
        padding: str = "max_length",
        num_workers: int = 0,
        DATA_FOLDER_PATH: str = DATA_FOLDER_PATH,
    ):
        assert language in ["tamil", "kannada", "malayalam"]
        super().__init__()
        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.max_length = max_length
        self.padding = padding
        self.language = language
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.DATA_FOLDER_PATH = DATA_FOLDER_PATH

    @staticmethod
    def _check_for_files(DATA_FOLDER_PATH: str, language: str) -> bool:
        train_file = Path(
            os.path.join(
                DATA_FOLDER_PATH, language, f"{language}_sentiment_full_train.tsv"
            )
        )
        val_file = Path(
            os.path.join(
                DATA_FOLDER_PATH, language, f"{language}_sentiment_full_dev.tsv"
            )
        )
        test_file = Path(
            os.path.join(
                DATA_FOLDER_PATH,
                language,
                f"{language}_sentiment_full_test_withlabels.tsv",
            )
        )

        return (
            (train_file.exists() and train_file.is_file())
            and (val_file.exists() and val_file.is_file())
            and (test_file.exists() and test_file.is_file())
        )

    def prepare_data(self):
        assert CodeMixingSentimentClassifierDataModule._check_for_files(
            self.DATA_FOLDER_PATH, self.language
        )

    def setup(self, stage):
        # Load the data
        datafiles = {
            "train": os.path.join(
                DATA_FOLDER_PATH,
                self.language,
                f"{self.language}_sentiment_full_train.tsv",
            ),
            "val": os.path.join(
                DATA_FOLDER_PATH,
                self.language,
                f"{self.language}_sentiment_full_dev.tsv",
            ),
            "test": os.path.join(
                DATA_FOLDER_PATH,
                self.language,
                f"{self.language}_sentiment_full_test_withlabels.tsv",
            ),
        }

        datasets: Dict[str, Dataset] = {
            "train": CodeMixingSentimentClassifierDataset(
                datafiles["train"], self.tokenizer, self.language, self.max_length
            ),
            "val": CodeMixingSentimentClassifierDataset(
                datafiles["val"], self.tokenizer, self.language, self.max_length
            ),
            "test": CodeMixingSentimentClassifierDataset(
                datafiles["test"], self.tokenizer, self.language, self.max_length
            ),
        }

        setattr(self, "datasets", datasets)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage):
        pass
