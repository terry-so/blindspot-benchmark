"""Config precedence and API dispatch without network calls."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import run


class CLITests(unittest.TestCase):
    def test_ten_percent_pilot_then_full_resume_without_scoring(self):
        base = json.loads(self.dataset.read_text())[0]
        cases = [base | {'case_id': f'{sub}_{i:02}', 'subcategory': sub,
                         'prompts': {'L1': 'red', 'L2b': 'red', 'L3': ['one', 'two']}}
                 for sub in ['1.1', '1.2'] for i in range(20)]
        self.dataset.write_text(json.dumps(cases))
        self.config.write_text('levels: [L1, L3-reset]\njudge: {provider: openai, model: judge}\nmodels:\n'
                               '  - {name: sunburst, provider: openai, model: sunburst}\n'
                               '  - {name: flare, provider: openai, model: flare}\n')
        editor, judge = Mock(), Mock()
        def edit(prompt, image, output):
            result = Image.new('RGB', (8, 8), 'red')
            result.save(output)
            return {'status': 'success', 'image': result, 'text_response': 'Done'}
        editor.generate_edit.side_effect = edit
        judge.detect_refusal.return_value = 'COMPLY'
        factory = Mock(return_value=editor)
        modules = {'models.wrappers': SimpleNamespace(AzureImageEditor=factory, OpenAIImageEditor=factory, GeminiImageEditor=factory),
                   'models.judges': SimpleNamespace(OpenAIJudge=Mock(return_value=judge))}
        args = self.args + ['--models', 'sunburst', '--run-percent', '10']
        with patch.dict('sys.modules', modules), patch.dict(os.environ, {'OPENAI_API_KEY': 'secret'}):
            self.assertEqual(run.main(args), 0)
            self.assertEqual(editor.generate_edit.call_count, 12)  # Four cases, one edit + two L3 steps.
            snapshot = self.root / 'run/data/dataset.json'
            self.assertEqual(len(json.loads(snapshot.read_text())), 40)
            self.assertFalse((self.root / 'run/results/flare').exists())
            self.assertEqual(run.main(args), 0)
            self.assertEqual(editor.generate_edit.call_count, 12)
            self.assertEqual(run.main(self.args + ['--models', 'sunburst']), 0)
            self.assertEqual(editor.generate_edit.call_count, 120)  # Pilot edits were not repeated.
        self.assertEqual(judge.detect_refusal.call_count, 80)
        judge.judge_fidelity.assert_not_called()
        judge.judge_realism.assert_not_called()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / 'input.png'
        Image.new('RGB', (16, 16)).save(image)
        self.dataset = self.root / 'cases.json'
        self.dataset.write_text(json.dumps([{
            'case_id': 'c1', 'domain': 'D1', 'subcategory': '1.1',
            'image_path': str(image), 'prompts': {'L1': 'red'},
            'verification': {'expected_result': 'red'},
        }]))
        self.config = self.root / 'config.yaml'
        self.config.write_text('provider: openai\nmodel: example\njudge:\n  provider: openai\n  model: judge\nlevels: [L1]\nrepeats: 2\n')
        self.args = ['--config', str(self.config), '--dataset', str(self.dataset),
                     '--output', str(self.root / 'run')]

    def test_overrides_and_unknown_keys(self):
        args = run.parse_args(self.args + ['--repeats', '3'])
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.judge_model, 'judge')
        self.config.write_text(self.config.read_text() + 'typo: true\n')
        with self.assertRaises(SystemExit):
            run.parse_args(self.args)

    def test_api_routing_check_resume_and_dataset_change(self):
        for provider in ['openai', 'azure', 'gemini', 'vertex']:
            with self.subTest(provider=provider):
                editor, judge = Mock(), Mock()
                modules = {
                    'models.wrappers': SimpleNamespace(AzureImageEditor=editor, OpenAIImageEditor=editor, GeminiImageEditor=editor),
                    'models.judges': SimpleNamespace(OpenAIJudge=judge),
                }
                args = self.args + ['--provider', provider, '--output', str(self.root / provider)]
                with patch.dict('sys.modules', modules), patch.dict(os.environ, {
                    'OPENAI_API_KEY': 'secret', 'Azure_API_KEY': 'secret',
                    'AZURE_OPENAI_BASE_URL': 'https://example.test/', 'GEMINI_API_KEY': 'secret',
                    'GOOGLE_CLOUD_PROJECT': 'project',
                }), patch('run.run_benchmark', return_value={'failed': 0, 'unavailable': 0}) as bench:
                    self.assertEqual(run.main(args + ['--check']), 0)
                    bench.assert_not_called()
                    self.assertFalse((self.root / provider).exists())
                    self.assertEqual(run.main(args), 0)
                    bench.call_args.kwargs['model_factories'][provider]()
                    if provider in ['gemini', 'vertex']:
                        editor.assert_called_with('example', vertexai=provider == 'vertex')
                    else:
                        editor.assert_called_with('example')
                    self.assertIsNone(bench.call_args.kwargs['judge_factory'])
                    self.assertTrue(bench.call_args.kwargs['generation_only'])
                    judge.assert_not_called()
                    self.assertEqual(run.main(args), 0)
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        run.main(args + ['--repeats', '3'])
                    settings = (self.root / provider / 'settings.json').read_text()
                    self.assertNotIn('secret', settings)
                    data = json.loads(self.dataset.read_text())
                    data[0]['prompts']['L1'] += ' changed'
                    self.dataset.write_text(json.dumps(data))
                    self.assertEqual(run.main(args), 0)  # Resume uses the saved source cases.

    def test_local_dispatch_and_boolean_override(self):
        self.config.write_text('provider: local\nmodel: flux-2-dev\ncpu_offload: true\njudge:\n  provider: azure\n  model: judge\n')
        with patch('run.run_local', return_value=0) as local:
            self.assertEqual(run.main(self.args + ['--no-cpu-offload', '--smoke-test']), 0)
            self.assertNotIn('--cpu-offload', local.call_args.args[0])
            self.assertIn('--smoke-test', local.call_args.args[0])

    def test_examples_parse(self):
        for path in Path('configs').glob('*.yaml'):
            run.parse_args(['--config', str(path)])

    def test_balanced_selection_and_config_override(self):
        base = json.loads(self.dataset.read_text())[0]
        cases = [base | {'case_id': name, 'subcategory': sub}
                 for sub in ['1.2', '1.1'] for name in [sub + '_03', sub + '_01', sub + '_02']]
        self.dataset.write_text(json.dumps(cases))
        self.config.write_text(self.config.read_text() + 'cases_per_subcategory: 2\n')
        args = run.parse_args(self.args)
        selected = run.load_cases(self.dataset, args.cases_per_subcategory)
        self.assertEqual([c['case_id'] for c in selected], ['1.1_01', '1.1_02', '1.2_01', '1.2_02'])
        self.assertEqual(len(run.load_cases(self.dataset)), 6)
        self.assertEqual(run.parse_args(self.args + ['--cases-per-subcategory', '1']).cases_per_subcategory, 1)
        with self.assertRaisesRegex(ValueError, 'requested 4'):
            run.load_cases(self.dataset, 4)
        with self.assertRaises(ValueError):
            run.load_cases(self.dataset, 0)

    def test_multiple_api_models_share_pool_and_resume_subsets(self):
        self.config.write_text('judge: {provider: openai, model: judge}\nlevels: [L1]\nmax_workers: 2\nmodels:\n  - {name: first, provider: openai, model: one}\n  - {name: second, provider: openai, model: two}\n')
        modules = {
            'models.wrappers': SimpleNamespace(AzureImageEditor=Mock(), OpenAIImageEditor=Mock(), GeminiImageEditor=Mock()),
            'models.judges': SimpleNamespace(OpenAIJudge=Mock()),
        }
        with patch.dict('sys.modules', modules), patch.dict(os.environ, {'OPENAI_API_KEY': 'secret'}), patch(
            'run.run_benchmark', return_value={'failed': 0, 'unavailable': 0}
        ) as bench:
            self.assertEqual(run.main(self.args), 0)
            self.assertEqual(set(bench.call_args.kwargs['model_factories']), {'first', 'second'})
            self.assertEqual(bench.call_args.kwargs['max_workers'], 2)
            self.assertEqual(bench.call_count, 1)
            self.assertEqual(run.main(self.args + ['--models', 'second']), 0)
            self.assertEqual(set(bench.call_args.kwargs['model_factories']), {'second'})
            self.config.write_text(self.config.read_text().replace('model: two', 'model: different'))
            with self.assertRaisesRegex(ValueError, 'changed'):
                run.main(self.args)

    def test_multiple_local_models_use_separate_processes(self):
        self.config.write_text('judge: {provider: azure, model: judge}\nmax_workers: 3\nmodels:\n  - {name: qwen, provider: local, model: qwen-image-2.1}\n  - {name: flux, provider: local, model: flux-2-dev, cpu_offload: true}\n')
        with patch('run.subprocess.run', return_value=SimpleNamespace(returncode=0)) as process:
            self.assertEqual(run.main(self.args), 0)
            calls = [c.args[0] for c in process.call_args_list]
            self.assertEqual(len(calls), 4)
            self.assertTrue(all('--check' in c for c in calls[:2]))
            self.assertTrue(all('--check' not in c for c in calls[2:]))
            self.assertIn('qwen-image-2.1', calls[2])
            self.assertIn('flux-2-dev', calls[3])
            self.assertIn('--cpu-offload', calls[3])
