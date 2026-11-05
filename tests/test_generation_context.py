from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image
from models.inference import generate_single_edit, generate_l3_chain
from utils.benchmark_runner import load_generation


class GenerationContextTests(unittest.TestCase):
    def test_single_l3_failure_and_recovery_keep_generation_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.png'
            Image.new('RGB', (8, 8)).save(source)
            case = {'image_path': str(source), 'prompts': {'L1': 'edit', 'L3': ['one', 'two']}}
            model = Mock()
            model.generation_context = {'provider': 'vertex', 'project': 'project-a', 'location': 'global'}
            def edit(prompt, original, output):
                image = Image.new('RGB', (8, 8))
                image.save(output)
                return {'status': 'success', 'image': image}
            model.generate_edit.side_effect = edit
            _, single = generate_single_edit(model, case, 'L1', root / 'single.png')
            _, chain = generate_l3_chain(model, None, case, root / 'chain.png', 0)
            self.assertEqual(single['generation_context']['project'], 'project-a')
            self.assertEqual([s['generation_context']['project'] for s in chain['steps']], ['project-a', 'project-a'])
            model.generation_context['project'] = 'project-b'
            _, recovered = load_generation(single, root / 'single.png')
            self.assertEqual(recovered['generation_context']['project'], 'project-a')
            model.generate_edit.side_effect = ValueError('failure')
            _, failed = generate_single_edit(model, case, 'L1', root / 'failed.png')
            self.assertEqual(failed['generation_context']['project'], 'project-b')
