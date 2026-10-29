import unittest

from estimate_image_cost import usage_cost


class ImageCostTests(unittest.TestCase):
    def test_separate_input_modalities_and_output(self):
        usage = {'input_tokens_details': {'text_tokens': 100, 'image_tokens': 200},
                 'output_tokens': 1000}
        self.assertAlmostEqual(usage_cost(usage, 5, 8, 30), 0.0321)

    def test_missing_usage_is_unknown(self):
        self.assertIsNone(usage_cost({}, 5, 8, 30))
        self.assertIsNone(usage_cost({'output_tokens': 100}, 5, 8, 30))

    def test_reported_zero_usage_is_zero(self):
        usage = {'input_tokens_details': {'text_tokens': 0, 'image_tokens': 0},
                 'output_tokens': 0}
        self.assertEqual(usage_cost(usage, 5, 8, 30), 0)
