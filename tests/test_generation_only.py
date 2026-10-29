import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from run_benchmark import run_benchmark


class GenerationOnlyTests(unittest.TestCase):
    def test_standalone_smoke_script(self):
        import runpy
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.png'
            Image.new('RGB', (8, 8)).save(source)
            case = {'case_id': 'c1', 'image_path': str(source), 'prompts': {'L1': 'red'}}
            editor = Mock()
            def edit(prompt, input_path, output_path):
                Image.new('RGB', (8, 8), 'red').save(output_path)
                return {'status': 'success'}
            editor.generate_edit.side_effect = edit
            api = Mock()
            api.model_info.return_value.sha = 'revision'
            modules = {
                'torch': SimpleNamespace(cuda=SimpleNamespace(
                    is_available=lambda: True, is_bf16_supported=lambda: True)),
                'huggingface_hub': SimpleNamespace(HfApi=lambda: api),
            }
            with patch.dict('sys.modules', modules), patch('run_local.load_cases', return_value=[case]), patch(
                'models.local_editors.LocalEditor', return_value=editor
            ), patch('sys.argv', ['test_local_generation.py', '--model', 'flux-2-dev', '--output', str(root / 'out')]):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'test_local_generation.py'), run_name='__main__')
            self.assertTrue((root / 'out/flux-2-dev/edited.png').is_file())
            editor.generate_edit.assert_called_once()

    def test_generation_chain_resume_and_no_grading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.png'
            Image.new('RGB', (8, 8), 'white').save(source)
            case = {'case_id': 'c1', 'domain': 'D1', 'subcategory': '1.1',
                    'image_path': str(source), 'prompts': {'L1': 'red', 'L3': ['red', 'blue']},
                    'verification': {'expected_result': 'blue'}}
            inputs = []

            def edit(prompt, input_path, output_path):
                with Image.open(input_path) as image:
                    inputs.append(image.getpixel((0, 0)))
                result = Image.new('RGB', (8, 8), prompt)
                result.save(output_path)
                return {'status': 'success', 'image': result}

            editor = Mock()
            editor.generate_edit.side_effect = edit
            options = dict(models={'local': editor}, generation_only=True,
                           levels=['L1', 'L3-reset'], result_dir=root / 'results', progress=False)
            with patch('run_benchmark.grade_output', side_effect=AssertionError('Judge called')):
                self.assertEqual(run_benchmark([case], **options)['successful'], 2)
                self.assertEqual(inputs[-1], (255, 0, 0))
                for path in (root / 'results/local').glob('*.json'):
                    result = json.loads(path.read_text())
                    self.assertEqual(result['status'], 'generated')
                    self.assertIsNone(result['r'])
                    self.assertIsNone(result['e'])
                    self.assertIsNone(result['q'])
                self.assertEqual(run_benchmark([case], **options)['skipped'], 2)
                self.assertEqual(editor.generate_edit.call_count, 3)

    def test_failed_chain_stops_without_refusal_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.png'
            Image.new('RGB', (8, 8)).save(source)
            case = {'case_id': 'c1', 'domain': 'D1', 'subcategory': '1.1',
                    'image_path': str(source), 'prompts': {'L3': ['one', 'two']},
                    'verification': {'expected_result': 'target'}}
            editor = Mock()
            editor.generate_edit.return_value = {'status': 'failed', 'image': None,
                                                'technical_error': True, 'error': 'OOM'}
            result = run_benchmark([case], models={'local': editor}, generation_only=True,
                                   levels=['L3-reset'], result_dir=root, progress=False)
            self.assertEqual(result['failed'], 1)
            self.assertEqual(editor.generate_edit.call_count, 1)
            saved = json.loads((root / 'local/c1_L3-reset_0.json').read_text())
            self.assertIsNone(saved['r'])
