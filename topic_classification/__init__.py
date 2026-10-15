"""Zero-shot topic classification for short English texts using GLiNER2."""
from .classifier import TopicClassifier, TopicResult
from .topics import DEFAULT_TOPICS

__all__ = ["TopicClassifier", "TopicResult", "DEFAULT_TOPICS"]
