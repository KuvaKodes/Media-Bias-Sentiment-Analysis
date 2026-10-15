import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from topic_classification import TopicClassifier
from topic_classification.__main__ import main


class TopicTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TopicClassifier()
        self.backend = Mock()
        self.backend.extract.return_value = {"topic": {"label": "gun issues", "confidence": 0.87}}
        self.classifier._model = self.backend
        self.classifier._schema = object()

    def test_result_and_schema_forwarding(self):
        result = self.classifier.classify("A mass shooting occurred.")
        self.assertEqual(result.topic, "gun issues")
        self.assertEqual(result.confidence, 0.87)
        self.backend.extract.assert_called_once_with(
            "A mass shooting occurred.", self.classifier._schema, include_confidence=True)

    def test_invalid_text_rejected_before_loading(self):
        with patch.object(self.classifier, "_load") as load:
            for value in (None, 1, "", "  ", "x" * 1001):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.classifier.classify(value)
            load.assert_not_called()

    def test_custom_topics_are_copied_and_validated(self):
        topics = {"sports": "Athletics", "other": "Everything else"}
        classifier = TopicClassifier(topics)
        topics["sports"] = "Changed"
        self.assertEqual(classifier._topics["sports"], "Athletics")
        for value in ({}, [], {"one": "Only one"}, {"a": "", "b": "B"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TopicClassifier(value)

    def test_invalid_response_is_not_reported_as_prediction(self):
        for value in ({}, {"topic": "gun issues"},
                      {"topic": {"label": "unexpected", "confidence": 0.9}},
                      {"topic": {"label": "gun issues", "confidence": float("nan")}},
                      {"topic": {"label": "gun issues", "confidence": 1.5}}):
            self.backend.extract.return_value = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.classifier.classify("Text")

    def test_all_cli_input_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("A mass shooting occurred.", encoding="utf-8")
            for args in (["--text", "A mass shooting occurred."], ["--file", str(path)], ["--stdin"]):
                output = io.StringIO()
                with patch("topic_classification.__main__.TopicClassifier", return_value=self.classifier), patch("sys.stdin", io.StringIO("A mass shooting occurred.")), contextlib.redirect_stdout(output):
                    main(args)
                self.assertEqual(json.loads(output.getvalue())["topic"], "gun issues")

    def test_malformed_topic_json_is_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topics.json"
            path.write_text("{invalid", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(["--text", "Text", "--topics", str(path)])
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
