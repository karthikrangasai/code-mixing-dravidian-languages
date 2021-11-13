import os
import pandas as pd
from copy import deepcopy
from code_mixing_dravidian_languages import DATA_FOLDER_PATH
from code_mixing_dravidian_languages.src.preprocess import _text_preprocess_fn

ind = 0
def _map(row):
    _row = deepcopy(row)
    _row["text"] = _text_preprocess_fn(_row["text"], language="tamil")
    global ind
    print(ind)
    ind += 1
    return _row

train = os.path.join(DATA_FOLDER_PATH, "fire_2020_sentiment/tamil/tamil_sentiment_full_train.tsv")
dev = os.path.join(DATA_FOLDER_PATH, "fire_2020_sentiment/tamil/tamil_sentiment_full_dev.tsv")

os.mkdir(os.path.join(DATA_FOLDER_PATH, "fire_2020_transliterated"))
os.mkdir(os.path.join(DATA_FOLDER_PATH, "fire_2020_transliterated", "tamil"))

new_train = os.path.join(DATA_FOLDER_PATH, "fire_2020_transliterated/tamil/tamil_sentiment_full_train.tsv")
new_dev = os.path.join(DATA_FOLDER_PATH, "fire_2020_transliterated/tamil/tamil_sentiment_full_dev.tsv")


ind = 0
train_df = pd.read_csv(train, sep="\t")
new_train_df = train_df.apply(_map, axis=1)
new_train_df.to_csv(new_train, sep="\t")


ind = 0
dev_df = pd.read_csv(train, sep="\t")
new_dev_df = dev_df.apply(_map, sep="\t")
new_train_df.to_csv(new_dev, sep="\t")
