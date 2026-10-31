import json
from pathlib import Path
import tempfile
import unittest

from utils.costs import estimate_cost, add_costs, write_cost_summary, validate_pricing


class CostTests(unittest.TestCase):
    rates = {'text_input': 5, 'image_input': 8, 'output': 30}
    usage = {'input_tokens': 1100, 'input_tokens_details': {'text_tokens': 100, 'image_tokens': 1000},
             'output_tokens': 2000}

    def test_usage_and_unknown_cost(self):
        self.assertAlmostEqual(estimate_cost(self.usage, self.rates), .0685)
        for usage in (None, {}, {'output_tokens': 2000},
                      self.usage | {'input_tokens_details': {'text_tokens': 100, 'image_tokens': 1000, 'cached_tokens': 10}}):
            self.assertIsNone(estimate_cost(usage, self.rates))
        self.assertIsNone(estimate_cost(self.usage, None))
        for rates in ({}, {'text_input': -1, 'image_input': 8, 'output': 30}, self.rates | {'output': float('nan')}):
            with self.assertRaises(ValueError):
                validate_pricing(rates)

    def test_l3_totals_and_resume_do_not_count_final_usage_twice(self):
        result = {'usage': self.usage, 'steps': [
            {'execution_status': 'success', 'usage': self.usage},
            {'execution_status': 'failed', 'safety_block': True}]}
        add_costs(result, self.rates)
        self.assertIsNone(result['estimated_cost_usd'])
        self.assertAlmostEqual(result['steps'][0]['estimated_cost_usd'], .0685)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'results' / 'sunburst'
            folder.mkdir(parents=True)
            (folder / 'case.json').write_text(json.dumps(result))
            (folder / 'case.generation.json').write_text(json.dumps(result))
            write_cost_summary(root / 'results')
            first = (root / 'cost_summary.json').read_bytes()
            write_cost_summary(root / 'results')
            self.assertEqual(first, (root / 'cost_summary.json').read_bytes())
            total = json.loads(first)['models']['sunburst']
            self.assertEqual(total, {'estimated_cost_usd': .0685, 'priced_calls': 1, 'unpriced_calls': 1})
        result['steps'][1]['usage'] = self.usage
        add_costs(result)  # Saved rates are reused.
        self.assertAlmostEqual(result['estimated_cost_usd'], .137)

    def test_runner_prices_edits_and_resumes_without_generation(self):
        from unittest.mock import Mock
        from PIL import Image
        from run_benchmark import run_benchmark
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.png'
            Image.new('RGB', (8, 8)).save(source)
            case = {'case_id': 'c1', 'domain': 'D1', 'subcategory': 'S1',
                    'image_path': str(source), 'prompts': {'L1': 'edit'}}
            model = Mock()
            def edit(prompt, original, output):
                image = Image.new('RGB', (8, 8))
                image.save(output)
                return {'status': 'success', 'image': image, 'usage': self.usage}
            model.generate_edit.side_effect = edit
            options = dict(cases=[case], models={'sunburst': model}, generation_only=True,
                           result_dir=root / 'results', progress=False, pricing={'sunburst': self.rates})
            run_benchmark(**options)
            result = json.loads((root / 'results/sunburst/c1_L1_0.json').read_text())
            self.assertEqual(result['estimated_cost_usd'], .0685)
            first = (root / 'cost_summary.json').read_bytes()
            run_benchmark(**options)
            model.generate_edit.assert_called_once()
            self.assertEqual(first, (root / 'cost_summary.json').read_bytes())
