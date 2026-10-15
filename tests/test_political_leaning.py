import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from political_leaning import PoliticalLeaningClassifier
from political_leaning.__main__ import main


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(return_value=[
            {"label": "LABEL_2", "score": 0.7},
            {"label": "LABEL_0", "score": 0.2},
            {"label": "LABEL_1", "score": 0.1},
        ])
        self.backend.tokenizer.return_value = {"input_ids": [1] * 12}
        self.classifier = PoliticalLeaningClassifier()
        self.classifier._pipeline = self.backend

    def test_maps_labels_independently_of_score_order(self):
        result = self.classifier.grade("Example political passage.")
        self.assertEqual(result.label, "right")
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.probabilities, {"left": 0.2, "center": 0.1, "right": 0.7})
        self.assertFalse(result.truncated)
        self.assertEqual(result.input_tokens, 12)

    def test_empty_and_invalid_input_does_not_load_model(self):
        classifier = PoliticalLeaningClassifier()
        with patch.object(classifier, "_load") as load:
            for text in ("", " \n", None, 123):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    classifier.grade(text)
            load.assert_not_called()

    def test_length_boundary_and_explicit_truncation(self):
        self.backend.tokenizer.return_value = {"input_ids": [1] * 512}
        self.assertFalse(self.classifier.grade("Text").truncated)
        self.backend.reset_mock()
        self.backend.tokenizer.return_value = {"input_ids": [1] * 513}
        with self.assertRaisesRegex(ValueError, "513 tokens"):
            self.classifier.grade("Text")
        self.backend.assert_not_called()
        result = self.classifier.grade("Text", truncate=True)
        self.assertTrue(result.truncated)
        self.backend.assert_called_once_with("Text", top_k=None, truncation=True, max_length=512)

    def test_rejects_changed_model_output(self):
        for scores in (
            [{"label": "unknown", "score": 1}],
            [{"label": "LABEL_0", "score": 1}],
            [{"label": "LABEL_0", "score": float("nan")}],
        ):
            with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                self.backend.return_value = scores
                self.classifier.grade("Text")

    def test_cli_sources_produce_json_without_echoing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passage.txt"
            path.write_text("A political passage", encoding="utf-8")
            for args in (["--text", "A political passage"], ["--file", str(path)], ["--stdin"]):
                output = io.StringIO()
                with patch("political_leaning.__main__.PoliticalLeaningClassifier", return_value=self.classifier), patch("sys.stdin", io.StringIO("A political passage")), contextlib.redirect_stdout(output):
                    main(args)
                self.assertEqual(json.loads(output.getvalue())["label"], "right")
                self.assertNotIn("A political passage", output.getvalue())

    def test_missing_file_exits_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--file", str(Path(directory) / "missing.txt")])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
