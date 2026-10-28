import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from data_foundry.generate_seed import generate_cases, update_reviewed_dataset
from run_local import load_cases


class ReviewedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reviewed = self.root / "data" / "benchmark_cases"
        self.folder = self.reviewed / "D1_identity" / "1.1_ids" / "case1"
        self.folder.mkdir(parents=True)
        self.case = {
            "case_id": "case1", "domain": "D1", "subcategory": "1.1",
            "image_path": "old/missing/input.png", "prompts": {"L1": "edit"},
            "verification": {"expected_result": "red"},
        }
        (self.folder / "case.json").write_text(json.dumps(self.case))
        Image.new("RGB", (16, 16), "red").save(self.folder / "input.png")

    def test_nested_folder_replaces_stale_index_and_image_paths(self):
        (self.reviewed / "dataset.json").write_text(json.dumps([{"case_id": "removed"}]))
        cases = load_cases(self.reviewed)
        self.assertEqual([c["case_id"] for c in cases], ["case1"])
        self.assertEqual(Path(cases[0]["image_path"]), self.folder / "input.png")
        self.assertEqual(json.loads((self.folder / "case.json").read_text()), self.case)

    def test_selection_copies_image_from_organized_folder(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            cases, run_dir = generate_cases(None, None, {"1.1": 1})
            with Image.open(cases[0]["image_path"]) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            self.assertTrue((run_dir / "data/dataset.json").is_file())
        finally:
            os.chdir(previous)

    def test_duplicate_ids_are_rejected(self):
        other = self.reviewed / "other" / "case1"
        other.mkdir(parents=True)
        (other / "case.json").write_text(json.dumps(self.case))
        Image.new("RGB", (16, 16)).save(other / "input.png")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            update_reviewed_dataset(self.reviewed)

    def test_missing_adjacent_image_is_rejected(self):
        (self.folder / "input.png").unlink()
        with self.assertRaises(FileNotFoundError):
            update_reviewed_dataset(self.reviewed)
