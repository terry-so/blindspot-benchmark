import csv
import json
from pathlib import Path
import tempfile
from threading import Barrier, Lock
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from evaluators.regrade import select_cases, regrade, agreement
from judge_results import main


class RegradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.labels = self.root / 'labels.csv'
        self.rows = []
        for i, (human, ai) in enumerate([('1', 0), ('0', 1), ('1', 1), ('0', 0), ('', 1), ('1', None)]):
            case_id = f'c{i}'
            data = self.run / 'data' / case_id
            data.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(data / 'input.png')
            case = {'case_id': case_id, 'prompts': {'L0': 'control', 'L1': 'edit', 'L2b': 'literal target'}}
            (data / 'case.json').write_text(json.dumps(case))
            level = 'L0' if i == 3 else 'L1'
            output = f'results/model/{case_id}_{level}_0.png'
            image = self.run / output
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8)).save(image)
            result = {'case_id': case_id, 'model': 'model', 'level': level, 'subcategory': '1.1',
                      'execution_status': 'success', 'text_response': 'Done', 'status': 'ok', 'r': 0 if ai is not None else 1, 'e': ai, 'q': 4 if ai == 1 else None}
            image.with_suffix('.json').write_text(json.dumps(result))
            self.rows.append({k: result[k] for k in ('case_id', 'model', 'level')} |
                             {'output_file': output, 'human_fidelity': human, 'human_refused': '0', 'human_realism': '5'})
        self.write_labels()

    def write_labels(self):
        with self.labels.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_selection_and_disagreements(self):
        selected, selection = select_cases(self.run, self.labels)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selection['missing_human'], 1)
        self.assertEqual(selection['missing_ai'], 1)
        self.assertEqual(selection['baseline_all_labeled'], {'n': 4, 'agreement': .5, 'kappa': 0, 'false_positives': 1, 'false_negatives': 1})
        self.assertEqual(selected[3]['target'], 'control')
        self.assertEqual(selected[0]['target'], 'literal target')
        selected, selection = select_cases(self.run, self.labels, True)
        self.assertEqual([r['case_id'] for r in selected], ['c0', 'c1'])
        self.assertEqual(selection['comparable'], 4)
        self.assertEqual(selection['selected'], 2)
        with self.assertRaisesRegex(ValueError, 'requires --labels'):
            select_cases(self.run, disagreements_only=True)

    def test_regrade_changes_no_source_and_reports_regressions(self):
        before = {str(p): p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        selected, selection = select_cases(self.run, self.labels)
        judge = Mock()
        with patch('evaluators.regrade.check_fidelity', side_effect=[1, 0, 0, 0]) as check:
            report = regrade(selected, self.root / 'new', judge, selection)
        self.assertEqual(check.call_count, 4)
        self.assertEqual(judge.mock_calls, [])  # No refusal or realism calls.
        self.assertEqual(report['after']['agreement'], .75)
        self.assertEqual(report['after']['kappa'], .5)
        self.assertEqual(report['disagreements_resolved'], 2)
        self.assertEqual(report['agreements_broken'], 1)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.run.rglob('*') if p.is_file()})
        self.assertEqual(len(json.loads((self.root / 'new/scores.json').read_text())), 4)
        self.assertTrue((self.root / 'new/scores.csv').is_file())
        self.assertTrue((self.root / 'new/fidelity_judge_prompt.txt').is_file())

    def test_failures_compare_identical_pairs(self):
        selected, selection = select_cases(self.run, self.labels)
        with patch('evaluators.regrade.check_fidelity', side_effect=[ValueError('offline'), 0, 1, 0]):
            report = regrade(selected, self.root / 'new', Mock(), selection)
        self.assertEqual(report['failed'], 1)
        self.assertEqual(report['before']['n'], 3)
        self.assertEqual(report['after']['n'], 3)
        self.assertEqual(report['after']['agreement'], 1)
        records = json.loads((self.root / 'new/scores.json').read_text())
        self.assertIsNone(records[0]['new_score'])
        self.assertEqual(records[0]['error'], 'offline')

    def test_disagreements_call_only_fidelity_with_saved_target(self):
        selected, selection = select_cases(self.run, self.labels, True)
        judge = Mock()
        judge.judge_fidelity.side_effect = [1, 0]
        report = regrade(selected, self.root / 'new', judge, selection)
        self.assertEqual(judge.judge_fidelity.call_count, 2)
        judge.detect_refusal.assert_not_called()
        judge.judge_realism.assert_not_called()
        self.assertIn('literal target', judge.judge_fidelity.call_args.args[2])
        self.assertEqual(report['before']['n'], 2)
        self.assertEqual(report['before']['agreement'], 0)
        self.assertEqual(report['after']['agreement'], 1)
        self.assertEqual(report['selection']['comparable'], 4)

    def test_check_requires_no_credentials_and_creates_no_output(self):
        with patch.dict('os.environ', {}, clear=True):
            code = main(['--run', str(self.run), '--output', str(self.root / 'new'),
                         '--metrics', 'fidelity', '--labels', str(self.labels), '--disagreements-only', '--check'])
        self.assertEqual(code, 0)
        self.assertFalse((self.root / 'new').exists())

    def test_invalid_labels_and_images_fail_before_calls(self):
        self.rows.append(self.rows[0])
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            select_cases(self.run, self.labels)
        self.rows.pop()
        self.rows[0]['output_file'] = '../outside.png'
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'inside'):
            select_cases(self.run, self.labels)

    def test_empty_and_constant_kappa(self):
        self.assertIsNone(agreement([], 'new_score')['kappa'])
        self.assertEqual(agreement([{'human_score': 1, 'new_score': 1}], 'new_score'),
                         {'n': 1, 'agreement': 1, 'kappa': None, 'false_positives': 0, 'false_negatives': 0})

    def test_parallel_workers_checkpoint_order_and_failure(self):
        selected, selection = select_cases(self.run, self.labels)
        barrier, lock = Barrier(2), Lock()
        active = peak = 0

        def check(judge, original, edited, target, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
                case = Path(original).parent.name
                if case == 'c1':
                    raise ValueError('invalid response')
                return 1 if case in ('c0', 'c2') else 0
            finally:
                with lock:
                    active -= 1

        with patch('evaluators.regrade.check_fidelity', side_effect=check):
            report = regrade(selected, self.root / 'parallel', Mock(), selection, max_workers=2)
        self.assertEqual(peak, 2)
        self.assertEqual(report['successful'], 3)
        self.assertEqual(report['failed'], 1)
        self.assertEqual(report['after']['n'], 3)
        self.assertEqual(report['after']['agreement'], 1)
        records = json.loads((self.root / 'parallel/scores.json').read_text())
        self.assertEqual([r['case_id'] for r in records], [r['case_id'] for r in selected])
        self.assertEqual(records[1]['error'], 'invalid response')
        self.assertIsNone(records[1]['new_score'])
        metadata = json.loads((self.root / 'parallel/judging.json').read_text())
        self.assertEqual(metadata['max_workers'], 2)

    def test_invalid_workers_do_not_create_output(self):
        selected, selection = select_cases(self.run, self.labels)
        for value in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                regrade(selected, self.root / 'invalid', Mock(), selection, max_workers=value)
        self.assertFalse((self.root / 'invalid').exists())

    def test_resume_reuses_successes_and_retries_failures(self):
        selected, selection = select_cases(self.run, self.labels)
        folder = self.root / 'resume'
        with patch('evaluators.regrade.check_fidelity', side_effect=[1, ValueError('failed'), 1, 0]):
            regrade(selected, folder, Mock(), selection)
        # Simulate an interrupted run: last case never reached its checkpoint.
        records = json.loads((folder / 'scores.json').read_text())
        (folder / 'scores.json').write_text(json.dumps(records[:3]))
        with patch('evaluators.regrade.check_fidelity', return_value=0) as check:
            report = regrade(selected, folder, Mock(), selection, resume=True, max_workers=2)
        self.assertEqual(check.call_count, 2)
        self.assertEqual(report['successful'], 4)
        with patch('evaluators.regrade.check_fidelity') as check:
            regrade(selected, folder, Mock(), selection, resume=True)
        check.assert_not_called()
        (folder / 'fidelity_judge_prompt.txt').write_text('different rubric')
        with patch('evaluators.regrade.check_fidelity') as check, self.assertRaisesRegex(ValueError, 'changed'):
            regrade(selected, folder, Mock(), selection, resume=True)
        check.assert_not_called()

    def test_locked_checkpoint_saves_recovery_and_resumes(self):
        from utils.benchmark_runner import atomic_json
        selected, selection = select_cases(self.run, self.labels)
        folder = self.root / 'locked'

        def write(path, value):
            if Path(path).name == 'scores.json':
                raise PermissionError('locked')
            atomic_json(path, value)

        with patch('evaluators.regrade.atomic_json', side_effect=write), \
             patch('evaluators.regrade.check_fidelity', return_value=1) as check, \
             self.assertRaisesRegex(RuntimeError, 'recovery'):
            regrade(selected, folder, Mock(), selection)
        saved = json.loads((folder / 'scores.recovery.json').read_text())
        self.assertEqual(len(saved), check.call_count)
        with patch('evaluators.regrade.check_fidelity', return_value=1) as check:
            report = regrade(selected, folder, Mock(), selection, resume=True)
        self.assertEqual(check.call_count, len(selected)-len(saved))
        self.assertEqual(report['successful'], 4)

    def test_realism_independent_of_new_fidelity_and_weighted_metrics(self):
        selected, selection = select_cases(self.run, self.labels, metric='realism')
        self.assertEqual(len(selected), 3)
        judge = Mock()
        judge.judge_realism.return_value = 5
        report = regrade(selected, self.root / 'realism', judge, selection)
        judge.judge_fidelity.assert_not_called()
        judge.detect_refusal.assert_not_called()
        self.assertEqual(report['before']['within_one'], 1)
        self.assertEqual(report['after']['agreement'], 1)
        self.assertIn('prompt', judge.judge_realism.call_args.kwargs)
        result = agreement([{'human_score': 1, 'new_score': 2},
                            {'human_score': 5, 'new_score': 4}], 'new_score', 'realism')
        self.assertAlmostEqual(result['quadratic_weighted_kappa'], 0.8)
        self.assertEqual(result['mean_absolute_difference'], 1)
        self.assertEqual(result['mean_ai_minus_human'], 0)

    def test_refusal_no_image_and_safety_block(self):
        path = self.run / self.rows[0]['output_file']
        path.unlink()
        saved = json.loads(path.with_suffix('.json').read_text())
        saved.update(r=1, image_returned=False, safety_block=True)
        path.with_suffix('.json').write_text(json.dumps(saved))
        selected, selection = select_cases(self.run, self.labels, True, 'refusal')
        self.assertEqual(len(selected), 2)
        judge = Mock()
        judge.detect_refusal.return_value = 'COMPLY'
        report = regrade(selected, self.root / 'refusal', judge, selection)
        self.assertEqual(report['before']['false_positives'], 2)
        self.assertEqual(report['after']['false_positives'], 1)
        judge.detect_refusal.assert_called_once()
        judge.judge_fidelity.assert_not_called()
        judge.judge_realism.assert_not_called()

    def test_l3_refusal_uses_observed_steps_without_changing_chain(self):
        path = (self.run / self.rows[0]['output_file']).with_suffix('.json')
        result = json.loads(path.read_text())
        result.update(level='L3-reset', r=1, chain_length=3, first_refusal_step=1,
                      steps=[{'step': 1, 'instruction': 'step one', 'execution_status': 'success',
                              'text_response': 'I cannot help', 'output_path': None, 'r': 1}])
        path.write_text(json.dumps(result))
        self.rows[0]['level'] = 'L3-reset'
        self.write_labels()
        before = path.read_bytes()
        selected, selection = select_cases(self.run, self.labels, True, 'refusal')
        with patch('evaluators.regrade.detect_refusal', return_value=0) as detect:
            regrade(selected, self.root / 'l3', Mock(), selection)
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(selected[0]['responses'][0]['instruction'], 'step one')

    def test_all_metrics_cli_and_resume(self):
        args = ['--run', str(self.run), '--output', str(self.root / 'calibration'),
                '--metrics', 'refusal', 'fidelity', 'realism', '--labels', str(self.labels),
                '--judge-model', 'test', '--max-workers', '2']
        judge = Mock()
        judge.detect_refusal.return_value = 'COMPLY'
        judge.judge_fidelity.return_value = 1
        judge.judge_realism.return_value = 5
        with patch('judge_results.create_judge', return_value=judge):
            self.assertEqual(main(args), 0)
            for metric in ('refusal', 'fidelity', 'realism'):
                self.assertTrue((self.root / 'calibration' / metric / 'agreement.json').is_file())
            judge.reset_mock()
            self.assertEqual(main(args + ['--resume']), 0)
            self.assertEqual(judge.mock_calls, [])

    def test_multi_metric_resume_after_first_metric(self):
        from evaluators.regrade import regrade as real_regrade
        args = ['--run', str(self.run), '--output', str(self.root / 'interrupted'),
                '--metrics', 'fidelity', 'realism', '--labels', str(self.labels), '--judge-model', 'test']
        judge = Mock()
        judge.judge_fidelity.return_value = 1
        judge.judge_realism.return_value = 5
        def interrupted(selected, destination, judge, selection, *args, **kwargs):
            if selection['metric'] == 'realism':
                raise RuntimeError('interrupted')
            return real_regrade(selected, destination, judge, selection, *args, **kwargs)
        with patch('judge_results.create_judge', return_value=judge):
            with patch('evaluators.regrade.regrade', side_effect=interrupted), self.assertRaisesRegex(RuntimeError, 'interrupted'):
                main(args)
            judge.reset_mock()
            self.assertEqual(main(args + ['--resume']), 0)
            judge.judge_fidelity.assert_not_called()
            self.assertEqual(judge.judge_realism.call_count, 3)

    def test_missing_refusal_metadata_fails_before_calls(self):
        path = (self.run / self.rows[0]['output_file']).with_suffix('.json')
        result = json.loads(path.read_text())
        del result['execution_status']
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'execution metadata'):
            select_cases(self.run, self.labels, metric='refusal')
