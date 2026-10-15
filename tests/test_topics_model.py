"""Opt-in smoke checks for the real GLiNER 2.0 checkpoint."""
import os
import unittest
from topic_classification import TopicClassifier


@unittest.skipUnless(os.environ.get("RUN_GLINER_MODEL_TESTS") == "1",
                     "Set RUN_GLINER_MODEL_TESTS=1 to download/load the model")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = TopicClassifier()

    def test_representative_topics(self):
        examples = [
            ("Donald Trump gave a speech at a rally today.", "president"),
            ("Another mass shooting has renewed calls for gun control.", "gun issues"),
            ("Transgender people deserve equal rights and protection from discrimination.", "gender rights"),
        ]
        for text, expected in examples:
            with self.subTest(expected=expected):
                result = self.classifier.classify(text)
                self.assertEqual(result.topic, expected)
                self.assertGreaterEqual(result.confidence, 0)
                self.assertLessEqual(result.confidence, 1)
