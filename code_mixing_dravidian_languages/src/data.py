import os
from types import LambdaType
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, default_data_collator

from code_mixing_dravidian_languages import DATA_FOLDER_PATH
from code_mixing_dravidian_languages.src.preprocess import (
    _text_preprocess_fn,
    _category_preprocess_fn,
)

DATA_METADATA = {
    "fire_2020":{
        "name": "FIRE_2020",
        "languages": ["all", "tamil", "malayalam", "kannada"],
        "data_folder_path": os.path.join(DATA_FOLDER_PATH, "fire_2020_sentiment"),
        "num_classes": 5,
        "category_mapping" : {
            "Positive ": 0,
            "Positive": 0,
            "Negative": 1,
            "not-Tamil": 2,
            "not-Kannada": 2,
            "not-malayalam": 2,
            "unknown_state": 3,
            "unknown state": 3,
            "Mixed feelings": 4,
            "Mixed_feelings": 4,
        },
        "filenames":{
            "train": lambda language: f"{language}_sentiment_full_train.tsv",
            "dev": lambda language: f"{language}_sentiment_full_dev.tsv",
            "test": lambda language: f"{language}_sentiment_full_test_withlabels.tsv",
        },
    },
    "codalab":{
        "name": "CODALAB_HATE_SPEECH",
        "languages": ["tamil"],
        "data_folder_path": os.path.join(DATA_FOLDER_PATH, "codalab_hate_speech"),
        "num_classes": 2,
        "category_mapping" : {
            "NOT": 0,
            "OFF": 1,
        },
        "filenames":{
            "train": lambda language: f"{language}_offensive_train.tsv",
            "dev": lambda language: f"{language}_offensive_dev.tsv",
        },
    }
}

class CodeMixingSentimentClassifierDataset(Dataset):
    def __init__(
        self, 
        filepath: str, 
        tokenizer: str, 
        language: str, 
        category_mapping: Dict[str, int],
        max_token_len: int = 128
    ):
        self.data = pd.read_csv(filepath, sep="\t")
        self.tokenizer = tokenizer
        self.language = language
        self.category_mapping = category_mapping
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        preprocessed_text = _text_preprocess_fn(data_row["text"], self.language)
        label = self.category_mapping[data_row["category"]]

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
        dataset: str,
        backbone: str = "ai4bharat/indic-bert",
        language: str = "tamil",
        batch_size: int = 8,
        max_length: int = 128,
        padding: str = "max_length",
        num_workers: int = 0,
    ):
        self.dataset_metadata = deepcopy(DATA_METADATA[dataset])
        assert language in self.dataset_metadata["languages"]
        super().__init__()
        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.max_length = max_length
        self.padding = padding
        self.language = language
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    @staticmethod
    def _check_for_files(data_folder_path: str, language: str, files: Dict[str, LambdaType]) -> bool:
        train_file = Path(os.path.join(data_folder_path, language, files["train"](language)))
        val_file = Path(os.path.join(data_folder_path, language, files["dev"](language)))
        return (train_file.exists() and train_file.is_file()) and (val_file.exists() and val_file.is_file())

    def prepare_data(self):
        data_folder_path = self.dataset_metadata["data_folder_path"]
        language = self.language
        files = self.dataset_metadata["filenames"]
        assert CodeMixingSentimentClassifierDataModule._check_for_files(
            data_folder_path=data_folder_path,
            language=language,
            files=files,
        )

    def setup(self, stage):
        # Load the data
        data_folder_path = self.dataset_metadata["data_folder_path"]
        language = self.language
        files = self.dataset_metadata["filenames"]
        datafiles = {
            "train": os.path.join(data_folder_path, language, files["train"](language)),
            "val": os.path.join(data_folder_path, language, files["dev"](language)),
        }

        self.datasets: Dict[str, Dataset] = {
            "train": CodeMixingSentimentClassifierDataset(
                filepath=datafiles["train"],
                tokenizer=self.tokenizer,
                language=self.language,
                max_token_len=self.max_length,
                category_mapping=self.dataset_metadata["category_mapping"],
            ),
            "val": CodeMixingSentimentClassifierDataset(
                filepath=datafiles["val"],
                tokenizer=self.tokenizer,
                language=self.language,
                max_token_len=self.max_length,
                category_mapping=self.dataset_metadata["category_mapping"],
            ),
        }

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

    def teardown(self, stage):
        pass
