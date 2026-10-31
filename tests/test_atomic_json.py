import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.benchmark_runner import atomic_json


class AtomicJsonTests(unittest.TestCase):
    def test_temporary_lock_retries_replace_without_rewriting_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scores.json'
            path.write_text('[0]')
            replace = os.replace
            calls = 0

            def locked(source, destination):
                nonlocal calls
                calls += 1
                if calls < 3:
                    self.assertEqual(json.loads(path.read_text()), [0])
                    raise PermissionError('sharing violation')
                replace(source, destination)

            with patch('utils.benchmark_runner.os.replace', side_effect=locked), \
                 patch('utils.benchmark_runner.time.sleep') as sleep:
                atomic_json(path, [1, 2])
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), [1, 2])
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_permanent_lock_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scores.json'
            path.write_text('[0]')
            with patch('utils.benchmark_runner.os.replace', side_effect=PermissionError('locked')) as replace, \
                 patch('utils.benchmark_runner.time.sleep'), self.assertRaises(PermissionError):
                atomic_json(path, [1])
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(json.loads(path.read_text()), [0])
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])
