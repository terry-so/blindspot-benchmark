import unittest
from unittest.mock import Mock
from evaluators.auto_grader import check_realism
from models.judges import OpenAIJudge


class RealismScaleTests(unittest.TestCase):
    def test_ordinal_score_and_reference_images(self):
        judge = object.__new__(OpenAIJudge)
        judge._ask = Mock(return_value='The pasted edges are obvious.\n<OUTPUT>2</OUTPUT>')
        details = {}
        score = check_realism(judge, 'b.png', details=details, edit_context='Change a date.', original_image_path='a.png')
        self.assertEqual(score, 2)
        self.assertEqual(judge._ask.call_args.args[1], ('a.png', 'b.png'))
        self.assertIn('Change a date.', judge._ask.call_args.args[0])
        self.assertNotIn('{edit context}', judge._ask.call_args.args[0])
        self.assertEqual(details['reason'], 'The pasted edges are obvious.')
        judge._ask.return_value = '<OUTPUT>0</OUTPUT>'
        with self.assertRaises(ValueError):
            judge.judge_realism_result('b.png', prompt='test')
