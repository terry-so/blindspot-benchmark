import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from PIL import Image
from judge_results import judge_results
from run_benchmark import run_benchmark


class SeparateJudgingTests(unittest.TestCase):
    def test_generate_then_judge_and_rejudge_without_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'run'
            data = source / 'data/c1'
            data.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(data / 'input.png')
            case = {'case_id': 'c1', 'domain': 'D1', 'subcategory': '1.1',
                    'image_path': str(data / 'input.png'),
                    'prompts': {'L1': 'red', 'L2b': 'make blue', 'L3': ['red', 'blue']},
                    'verification': {'expected_result': 'blue'}}
            (data / 'case.json').write_text(json.dumps(case))
            def edit(prompt, image, output):
                result = Image.new('RGB', (8, 8), prompt)
                result.save(output)
                return {'status': 'success', 'image': result, 'text_response': 'Done'}
            model = Mock()
            model.generate_edit.side_effect = edit
            judge = Mock()
            judge.detect_refusal.return_value = 'COMPLY'
            judge.judge_fidelity.return_value = 1
            judge.judge_realism.return_value = 4
            run_benchmark([case], models={'model': model}, judge=judge, generation_only=True,
                          levels=['L1', 'L3-reset'], result_dir=source / 'results', progress=False)
            self.assertEqual(judge.detect_refusal.call_count, 2)
            judge.judge_fidelity.assert_not_called()
            judge.judge_realism.assert_not_called()
            self.assertEqual(judge_results(source, root / 'scored', judge), 0)
            self.assertEqual(judge.judge_fidelity.call_count, 2)
            self.assertEqual(model.generate_edit.call_count, 3)

            self.assertEqual(json.loads((source / 'results/model/c1_L1_0.json').read_text())['status'], 'generated')
            judge.judge_fidelity.return_value = 0
            self.assertEqual(judge_results(root / 'scored', root / 'rescored', judge), 0)
            result = json.loads((root / 'rescored/results/model/c1_L1_0.json').read_text())
            self.assertEqual(result['e'], 0)
            self.assertIsNone(result['q'])
            self.assertEqual(model.generate_edit.call_count, 3)

    def test_text_only_response_is_saved_resumed_and_judged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'run/data/c1'
            data.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(data / 'input.png')
            case = {'case_id': 'c1', 'domain': 'D1', 'subcategory': '1.1',
                    'image_path': str(data / 'input.png'), 'prompts': {'L1': 'edit'},
                    'verification': {'expected_result': 'target'}}
            (data / 'case.json').write_text(json.dumps(case))
            model, judge = Mock(), Mock()
            model.generate_edit.return_value = {'status': 'success', 'image': None,
                                                'text_response': 'I cannot do that'}
            judge.detect_refusal.return_value = 'REFUSE'
            options = dict(models={'model': model}, judge=judge, generation_only=True,
                           levels=['L1'], result_dir=root / 'run/results', progress=False)
            run_benchmark([case], **options)
            self.assertEqual(run_benchmark([case], **options)['skipped'], 1)
            judge.detect_refusal.assert_not_called()
            self.assertEqual(judge_results(root / 'run', root / 'scored', judge), 0)
            result = json.loads((root / 'scored/results/model/c1_L1_0.json').read_text())
            self.assertEqual(result['r'], 1)
            judge.judge_fidelity.assert_not_called()

    def test_l3_refusal_stops_without_fidelity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (8, 8)).save(root / 'input.png')
            case = {'case_id': 'c1', 'domain': 'D1', 'subcategory': '1.1',
                    'image_path': str(root / 'input.png'), 'prompts': {'L3': ['one', 'two']},
                    'verification': {'expected_result': 'target'}}
            model, judge = Mock(), Mock()
            model.generate_edit.return_value = {'status': 'success', 'image': None, 'text_response': 'No'}
            judge.detect_refusal.return_value = 'REFUSE'
            summary = run_benchmark([case], models={'model': model}, judge=judge,
                                    generation_only=True, levels=['L3-reset'], result_dir=root, progress=False)
            self.assertEqual(summary['refused'], 1)
            self.assertEqual(model.generate_edit.call_count, 1)
            judge.judge_fidelity.assert_not_called()
