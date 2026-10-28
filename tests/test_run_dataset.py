import json
from pathlib import Path
import shutil
import tempfile
import unittest
from PIL import Image
from utils.run_dataset import run_cases


class SnapshotTests(unittest.TestCase):
    def test_copy_resume_move_and_selection_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'source.png'
            Image.new('RGB', (8, 8), 'red').save(image)
            dataset = root / 'source.json'
            dataset.write_text(json.dumps([{'case_id': 'c1', 'subcategory': '1.1',
                                           'image_path': str(image), 'prompts': {'L1': 'red'},
                                           'verification': {'expected_result': 'red'}}]))
            output = root / 'run'
            run_cases(output, dataset)
            self.assertFalse(output.exists())
            cases = run_cases(output, dataset, save=True)
            self.assertEqual(Path(cases[0]['image_path']).read_bytes(), image.read_bytes())
            self.assertTrue((output / 'data/c1/case.json').is_file())
            image.unlink()
            dataset.unlink()
            self.assertEqual(run_cases(output, dataset), cases)
            moved = root / 'moved'
            shutil.copytree(output, moved)
            self.assertTrue(Path(run_cases(moved, dataset)[0]['image_path']).is_file())
            with self.assertRaisesRegex(ValueError, 'selection changed'):
                run_cases(output, dataset, smoke=True)
            (output / 'data/c1/input.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Saved dataset changed'):
                run_cases(output, dataset)
