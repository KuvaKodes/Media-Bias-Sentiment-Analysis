import unittest
from unittest.mock import Mock
from political_leaning import PoliticalLeaningClassifier
from topic_classification import TopicClassifier


class BatchTests(unittest.TestCase):
    def test_political_order_and_truncation_metadata(self):
        model = Mock(return_value=[
            [{'label': 'LABEL_0', 'score': .8}, {'label': 'LABEL_1', 'score': .1}, {'label': 'LABEL_2', 'score': .1}],
            [{'label': 'LABEL_0', 'score': .1}, {'label': 'LABEL_1', 'score': .2}, {'label': 'LABEL_2', 'score': .7}],
        ])
        model.tokenizer.return_value = {'input_ids': [[1]*5, [1]*513]}
        classifier = PoliticalLeaningClassifier()
        classifier._pipeline = model
        with self.assertRaises(ValueError): classifier.grade_many(['short', 'long'])
        model.assert_not_called()
        results = classifier.grade_many(['short', 'long'], truncate=True)
        self.assertEqual([r.label for r in results], ['left', 'right'])
        self.assertEqual([r.truncated for r in results], [False, True])

    def test_topic_batch_order_and_invalid_input(self):
        classifier = TopicClassifier()
        classifier._model = Mock()
        classifier._model.batch_extract.return_value = [
            {'topic': {'label': 'president', 'confidence': .8}},
            {'topic': {'label': 'gun issues', 'confidence': .9}}]
        results = classifier.classify_many(['Trump', 'Gun control'])
        self.assertEqual([r.topic for r in results], ['president', 'gun issues'])
        with self.assertRaises(ValueError): classifier.classify_many(['x'*1001])
        classifier._model.batch_extract.return_value = []
        with self.assertRaises(RuntimeError): classifier.classify_many(['Trump'])

    def test_empty_batch_does_not_load_models(self):
        self.assertEqual(PoliticalLeaningClassifier().grade_many([]), [])
        self.assertEqual(TopicClassifier().classify_many([]), [])
