import os

code_mixing_dravidian_languages_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(code_mixing_dravidian_languages_ROOT_PATH)

DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, "data")

TODO = [
    "Combine Datasets",
    "Preprocess Dataset",
    "Test Data Pipeline",
    "Test Model Creation",
    "Test Trainer",
    "Add Finetuning Callbacks",
    "Create configurations - pretrain, finetune",
    "Create configurations - sweeps",
    "Create Sweep Script",
    "Create Training scripts",
]
