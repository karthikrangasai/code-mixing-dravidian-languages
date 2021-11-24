from code_mixing_dravidian_languages.src.models.hf_model import CodeMixingHFSentimentClassifier
from code_mixing_dravidian_languages.src.models.custom_model import CodeMixingCustomSentimentClassifier

MODEL_MAPPING = {
    "hf": CodeMixingHFSentimentClassifier,
    "custom": CodeMixingCustomSentimentClassifier,
}
