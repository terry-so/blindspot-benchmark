from collections import Counter
import unittest
from utils.run_dataset import execution_cases


class ExecutionCasesTests(unittest.TestCase):
    def test_balanced_reproducible_ten_percent(self):
        cases = [{'case_id': f'{sub}_{i:02}', 'subcategory': str(sub)}
                 for sub in range(30) for i in range(20)]
        selected = execution_cases(cases, 10)
        self.assertEqual(len(selected), 60)
        self.assertEqual(set(Counter(c['subcategory'] for c in selected).values()), {2})
        self.assertEqual(selected, execution_cases(list(reversed(cases)), 10))
        self.assertIs(execution_cases(cases, 100), cases)
        self.assertEqual(len(cases), 600)

    def test_invalid_limits_and_small_groups(self):
        for percent in (0, 101, True, 10.5):
            with self.assertRaises(ValueError):
                execution_cases([], percent)
        self.assertEqual(len(execution_cases([{'case_id': 'one', 'subcategory': '1'}], 10)), 1)
