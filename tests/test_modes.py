import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run


class ModeTests(unittest.TestCase):
    def test_modes_dispatch_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / 'run')
            base = ['--output', output, '--judge-model', 'judge']
            with patch('run.generate', return_value=0) as generate, patch('judge_results.main', return_value=0) as judge:
                run.main(base + ['--mode', 'judge'])
                generate.assert_not_called()
                self.assertIn(output + '-scored', judge.call_args.args[0])
                judge.reset_mock()
                run.main(base + ['--provider', 'openai', '--model', 'editor'])
                generate.assert_called_once()
                judge.assert_not_called()
            events = []
            with patch('run.generate', side_effect=lambda a: events.append('generate') or 0), \
                 patch('judge_results.main', side_effect=lambda a: events.append('judge') or 0), \
                 patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
                run.main(base + ['--mode', 'all', '--provider', 'openai', '--model', 'editor', '--judge-provider', 'openai'])
            self.assertEqual(events, ['generate', 'judge'])

    def test_failed_generation_and_check_do_not_judge(self):
        with tempfile.TemporaryDirectory() as folder:
            args = ['--output', str(Path(folder) / 'run'), '--mode', 'all', '--provider', 'openai',
                    '--model', 'editor', '--judge-provider', 'openai', '--judge-model', 'judge']
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                 patch('run.generate', return_value=1), patch('judge_results.main') as judge:
                self.assertEqual(run.main(args), 1)
                judge.assert_not_called()
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                 patch('run.generate', return_value=0), patch('judge_results.main') as judge:
                self.assertEqual(run.main(args + ['--check']), 0)
                judge.assert_not_called()
