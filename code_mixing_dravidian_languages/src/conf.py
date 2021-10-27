from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CodeMixingSentimentClassifierConfiguration:
    # Fixed values
    num_classes: int = field(init=False, default=5, repr=False)

    # Configurable values
    learning_rate: float = field(default=1e-5)
    batch_size: int = field(default=8)
    num_workers: int = field(default=0)
    max_epochs: int = field(default=10)
    operation_type: str = field(default="train")

    finetuning_strategy: str = field(default=None)

    backbone: str = field(default="ai4bharat/indic-bert")
    language: str = field(default="tamil")

    max_length: int = field(default=256)

    def __post_init__(self):
        assert self.operation_type in ["train", "finetune", "hparams_search"]
        assert self.language in ["tamil", "kannada", "malayalam"]


@dataclass
class WANDBLoggerConfiguration:
    group: str
    job_type: str
    name: str
    id: str
    config: Dict[str, Any]
