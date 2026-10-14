"""Validate experiment inputs without constructing API clients."""

import unittest

import test_benchmark as fixtures
from data_foundry.generate_seed import generate_cases, validate_categories
from utils.benchmark_runner import validate_inputs


class ValidationTests(unittest.TestCase):
    setUp = fixtures.BenchmarkTests.setUp

    def validate(self, **changes):
        options = dict(
            cases=[self.case],
            names=["editor"],
            levels=["L1"],
            repeats=1,
            max_workers=1,
            threshold=0.01,
        )
        options.update(changes)
        validate_inputs(**options)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (-1, float("nan"), float("inf"), True, "0.01"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.validate(threshold=threshold)
        self.validate(threshold=0)

    def test_names_are_simple_path_components(self):
        self.validate(names=["editor version 1"])
        for name in ("", " ", "..", "../outside", "folder/editor", "folder\\editor"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(names=[name])
        with self.assertRaises(ValueError):
            self.validate(names=["editor", "editor"])

    def test_l3_requires_nonempty_prompt_strings(self):
        for prompts in ([], "one prompt", [""], ["   "], [1]):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                self.case["prompts"]["L3"] = prompts
                self.validate(levels=["L3-reset"])

    def test_category_checks_belong_to_preparation(self):
        validate_categories([self.case])
        for changes in ({"domain": "D7"}, {"subcategory": "2.1"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_categories([self.case | changes])

    def test_invalid_dataset_requests_fail_before_generation(self):
        for counts in ({"7.1": 1}, {"1.1": -1}, {"1.1": 1.5}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                generate_cases(None, None, counts, use_reviewed=False)


if __name__ == "__main__":
    unittest.main()
