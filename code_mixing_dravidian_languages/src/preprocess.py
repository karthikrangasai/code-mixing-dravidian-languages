import re
import nltk
import string

from typing import Any, Dict

from indic_transliteration import sanscript
from google.transliteration import transliterate_word

from nltk.corpus import stopwords
nltk.download("stopwords")

_transliteration_map = {
    "tamil": sanscript.TAMIL,
    "kannada": sanscript.KANNADA,
    "malayalam": sanscript.MALAYALAM,
}

_google_transliteration_map = {
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
}


def _clean_text(text):
    review = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", str(text))
    review = re.sub(r"\([\s\S]*\)", " ", str(review))
    review = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", str(review))
    review = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", str(review))
    review = review.lower()
    review = re.sub(r"#(\w+)", "", str(review))
    review = re.sub(r"that's", "that is", str(review))
    review = re.sub(r"there's", "there is", str(review))
    review = re.sub(r"what's", "what is", str(review))
    review = re.sub(r"where's", "where is", str(review))
    review = re.sub(r"it's", "it is", str(review))
    review = re.sub(r"who's", "who is", str(review))
    review = re.sub(r"i'm", "i am", str(review))
    review = re.sub(r"she's", "she is", str(review))
    review = re.sub(r"he's", "he is", str(review))
    review = re.sub(r"they're", "they are", str(review))
    review = re.sub(r"who're", "who are", str(review))
    review = re.sub(r"ain't", "am not", str(review))
    review = re.sub(r"wouldn't", "would not", str(review))
    review = re.sub(r"shouldn't", "should not", str(review))
    review = re.sub(r"can't", "can not", str(review))
    review = re.sub(r"couldn't", "could not", str(review))
    review = re.sub(r"won't", "will not", str(review))
    review = re.sub(r" pm ", " ", str(review))
    review = re.sub(r" am ", " ", str(review))
    review = re.sub(r"[^\[\]]+(?=\])", " ", str(review))
    review = re.sub(r"\W", " ", str(review))
    review = re.sub(r"\d", " ", str(review))
    review = re.sub(r"\s+[a-z]\s+", " ", str(review))
    review = re.sub(r"\s+[a-z]$", " ", str(review))
    review = re.sub(r"^[a-z]\s+", " ", str(review))
    review = re.sub(r"\s+", " ", str(review))
    return review


def _transliterate_text(text, language: str):
    assert language in ["tamil", "kannada", "malayalam"]
    return sanscript.transliterate(text, sanscript.ITRANS, _transliteration_map[language])


def _google_transliteration_api_text(text: str, language: str):
    assert language in ["tamil", "kannada", "malayalam"]
    transliterated_tokens = []
    for t in text.split():
        _tokens = transliterate_word(t, lang_code=_google_transliteration_map[language])
        if len(_tokens) > 0:
            transliterated_tokens.append(_tokens[0])
    return " ".join(transliterated_tokens)

def _remove_punctuations(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def _remove_emoticons(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)

def _remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(text)


_preprocess_fn_mapping = {
    "indic": _transliterate_text,
    "google": _google_transliteration_api_text,
}

def _text_preprocess_fn(text, language: str = "tamil", preprocess_fn: str = None):
    # print(f"Before Preprocessing: {text}")
    text = _clean_text(text)
    # text = _remove_punctuations(text)
    text = _remove_emoticons(text)
    if preprocess_fn != None:
        text = _preprocess_fn_mapping[preprocess_fn](text, language=language)
    # text = _remove_stopwords(text)
    # text = _lemmatizate(text)
    return text


_category_mapping = {
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
}


def _category_preprocess_fn(category: str) -> int:
    return _category_mapping[category]


def preprocess_fn(sample: Dict[str, Any], language: str) -> Dict[str, Any]:
    sample["input"] = _text_preprocess_fn(sample.pop("text"), language=language)
    sample["label"] = _category_preprocess_fn(sample.pop("category"))
    return sample
