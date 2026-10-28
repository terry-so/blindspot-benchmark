import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from utils.notebook_review import show_edit, analyse_run


class NotebookReviewTests(unittest.TestCase):
    def test_review_uses_snapshot_and_saved_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case_dir = root / 'data/c1'
            case_dir.mkdir(parents=True)
            Image.new('RGB', (8, 8), 'white').save(case_dir / 'input.png')
            case = {'case_id': 'c1', 'prompts': {'L1': 'red'}}
            (case_dir / 'case.json').write_text(json.dumps(case))
            (root / 'data/dataset.json').write_text(json.dumps([case]))
            result_dir = root / 'results/flare'
            result_dir.mkdir(parents=True)
            Image.new('RGB', (8, 8), 'red').save(result_dir / 'c1.png')
            (result_dir / 'c1.json').write_text(json.dumps({
                'case_id': 'c1', 'model': 'flare', 'level': 'L1', 'status': 'ok', 'r': 0, 'e': 1, 'q': 4}))
            settings_dir = root / 'models/flare'
            settings_dir.mkdir(parents=True)
            (settings_dir / 'settings.json').write_text(json.dumps({'levels': ['L0', 'L1'], 'repeats': 2}))
            axes = [Mock(), Mock()]
            pyplot = Mock()
            pyplot.subplots.return_value = (Mock(), axes)
            tables = Mock(return_value={'overall': [], 'paired_levels': [], 'l3_summary': []})
            plots = Mock()
            with patch.dict('sys.modules', {
                'matplotlib': SimpleNamespace(pyplot=pyplot), 'matplotlib.pyplot': pyplot,
                'IPython.display': SimpleNamespace(display=Mock()),
                'utils.export_table': SimpleNamespace(make_analysis_tables=tables),
                'utils.plot_results': SimpleNamespace(make_plots=plots),
            }):
                show_edit(root)
                self.assertEqual(axes[0].imshow.call_args.args[0].getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(axes[1].imshow.call_args.args[0].getpixel((0, 0)), (255, 0, 0))
                analyse_run(root)
                self.assertEqual(tables.call_args.kwargs['levels'], ['L1'])
                self.assertEqual(tables.call_args.kwargs['repeats'], 2)
                self.assertEqual(tables.call_args.kwargs['cases'], [case])
                plots.assert_called_once()
