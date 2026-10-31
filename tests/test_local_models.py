"""Local GPU adapter and launcher checks without weights, GPUs or API calls."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from evaluators.auto_grader import detect_refusal
from models.local_editors import LocalEditor, edit_seed
from run_local import load_cases
from run_local import main
from run_benchmark import run_benchmark


class LocalModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.png"
        Image.new("RGB", (32, 32), "white").save(self.source)
        self.target = self.root / "qwen-image-2.1" / "case_L1_0.png"
        self.image = Image.new("RGB", (32, 32), "red")

    def editor(self, model="qwen-image-2.1"):
        editor = LocalEditor.__new__(LocalEditor)
        editor.model = model
        editor.model_name = model
        editor.seed = 42
        editor.cpu_offload = False
        editor.torch = SimpleNamespace(
            inference_mode=nullcontext,
            Generator=Mock(return_value=SimpleNamespace(manual_seed=lambda seed: seed)),
        )
        editor.pipe = Mock(return_value=SimpleNamespace(images=[self.image]))
        return editor

    def test_qwen_saves_image_and_seed(self):
        editor = self.editor()
        result = editor.generate_edit("make red", self.source, self.target)
        self.assertEqual(result["status"], "success")
        with Image.open(self.target) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        metadata = json.loads(self.target.with_suffix(".generation.json").read_text())
        self.assertEqual(metadata["instruction"], "make red")
        self.assertEqual(metadata["seed"], edit_seed(42, self.target))
        self.assertEqual(editor.pipe.call_args.kwargs["num_inference_steps"], 40)

    def test_flux_uses_and_records_expanded_prompt(self):
        editor = self.editor("flux-3-klein")
        editor._expand = Mock(return_value="A complete red scene")
        editor.pipe.return_value = [self.image]
        result = editor.generate_edit("make red", self.source, self.target)
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["text_response"])
        self.assertEqual(editor.pipe.call_args.kwargs["prompt"], "A complete red scene")
        self.assertEqual(editor.pipe.call_args.kwargs["num_inference_steps"], 16)
        metadata = json.loads(self.target.with_suffix(".generation.json").read_text())
        self.assertEqual(metadata["expanded_prompt"], "A complete red scene")

    def test_flux2_dev_edits_without_prompt_expansion(self):
        editor = self.editor("flux-2-dev")
        editor._expand = Mock()
        result = editor.generate_edit("make red", self.source, self.target)
        self.assertEqual(result["status"], "success")
        editor._expand.assert_not_called()
        args = editor.pipe.call_args.kwargs
        self.assertEqual(args["prompt"], "make red")
        self.assertEqual(args["image"].size, (32, 32))
        self.assertEqual(args["num_inference_steps"], 50)
        self.assertEqual(args["guidance_scale"], 4.0)
        editor.torch.Generator.assert_called_once_with("cpu")
        self.assertTrue(self.target.is_file())

    def test_flux2_dev_loads_revision_and_offload(self):
        for offload in (False, True):
            with self.subTest(cpu_offload=offload):
                pipeline = Mock()
                with patch.dict("sys.modules", {
                    "torch": SimpleNamespace(bfloat16="bf16"),
                    "diffusers": SimpleNamespace(Flux2Pipeline=pipeline),
                }):
                    LocalEditor("flux-2-dev", "fixed-revision", cpu_offload=offload)
                pipeline.from_pretrained.assert_called_once_with(
                    "black-forest-labs/FLUX.2-dev", revision="fixed-revision", torch_dtype="bf16")
                pipe = pipeline.from_pretrained.return_value
                if offload:
                    pipe.enable_model_cpu_offload.assert_called_once_with()
                    pipe.to.assert_not_called()
                else:
                    pipe.to.assert_called_once_with("cuda")

    def test_local_oom_is_error_without_judge_call(self):
        editor = self.editor()
        editor.pipe.side_effect = RuntimeError("CUDA out of memory")
        output = editor.generate_edit("edit", self.source, self.target)
        judge = Mock()
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            detect_refusal(judge, str(self.source), output, 0, "edit")
        judge.detect_refusal.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_prompt_expansion_error_never_calls_image_pipeline(self):
        editor = self.editor("flux-3-klein")
        editor._expand = Mock(side_effect=ValueError("empty caption"))
        output = editor.generate_edit("edit", self.source, self.target)
        self.assertTrue(output["technical_error"])
        editor.pipe.assert_not_called()
        metadata = json.loads(self.target.with_suffix(".generation.json").read_text())
        self.assertEqual(metadata["failed_stage"], "prompt_expansion")

    def test_seeds_stable_across_roots_but_different_for_repeats_and_steps(self):
        self.assertEqual(edit_seed(42, "/first/qwen/c1_L1_0.png"), edit_seed(42, "/second/qwen/c1_L1_0.png"))
        paths = ["qwen/c1_L1_0.png", "qwen/c1_L1_1.png",
                 "c1_L3-reset_0_steps/step_1.png", "c1_L3-reset_0_steps/step_2.png"]
        self.assertEqual(len({edit_seed(42, path) for path in paths}), 4)

    def test_windows_relative_dataset_paths_on_linux(self):
        folder = self.root / "images"
        folder.mkdir()
        self.image.save(folder / "input.png")
        dataset = self.root / "dataset.json"
        dataset.write_text(json.dumps([{"image_path": "images\\input.png"}]))
        with patch("run_local.ROOT", self.root / "elsewhere"):
            cases = load_cases(dataset)
        self.assertEqual(Path(cases[0]["image_path"]), folder / "input.png")

    def test_l3_and_resume_with_actual_local_adapter(self):
        editor = self.editor()
        judge = Mock()
        judge.judge_fidelity.return_value = 1
        judge.judge_realism.return_value = 4
        case = {
            "case_id": "c1", "domain": "D1", "subcategory": "1.1",
            "image_path": str(self.source), "prompts": {"L2b": "red", "L3": ["one", "two"]},
            "verification": {"expected_result": "red"},
        }
        kwargs = dict(models={"qwen-image-2.1": editor}, judge=judge, levels=["L3-reset"],
                      result_dir=self.root / "results", progress=False)
        self.assertEqual(run_benchmark([case], **kwargs)["successful"], 1)
        self.assertEqual(editor.pipe.call_count, 2)
        second_input = editor.pipe.call_args.kwargs["image"]
        self.assertEqual(second_input.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(run_benchmark([case], **kwargs)["skipped"], 1)
        self.assertEqual(editor.pipe.call_count, 2)

    def test_launcher_smoke_preflight_and_changed_settings(self):
        dataset = self.root / "dataset.json"
        case = {
            "case_id": "c1", "domain": "D1", "subcategory": "1.1",
            "image_path": str(self.source), "prompts": {"L1": "red"},
            "verification": {"expected_result": "red"},
        }
        dataset.write_text(json.dumps([case, case | {"case_id": "c2"}]))
        output = self.root / "run"
        args = ["--model", "qwen-image-2.1", "--dataset", str(dataset), "--output", str(output), "--smoke-test"]
        api = Mock()
        api.model_info.return_value.sha = "fixed-revision"
        judge = Mock()
        judge.detect_refusal.return_value = "COMPLY"
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True, is_bf16_supported=lambda: True,
                get_device_name=lambda index: "fake GPU")),
            "diffusers": SimpleNamespace(QwenImage21Pipeline=object, Flux2Pipeline=object),
            "huggingface_hub": SimpleNamespace(HfApi=lambda: api),
            "models.wrappers": SimpleNamespace(OpenAIJudge=lambda name: judge),
        }
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {
            "JUDGE_MODEL": "test-judge", "Azure_API_KEY": "test-secret",
            "AZURE_OPENAI_BASE_URL": "https://example.test/openai/v1/",
        }), patch("run_local.version", return_value="test-version"), patch(
            "run_local.subprocess.check_output", return_value="test==1\n"
        ), patch("run_local.LocalEditor") as editor, patch(
            "run_local.run_benchmark", return_value={"failed": 0, "unavailable": 0}
        ) as benchmark:
            self.assertEqual(main(args + ["--check"]), 0)
            editor.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(main(args), 0)
            self.assertEqual(len(benchmark.call_args.args[0]), 1)
            self.assertEqual(benchmark.call_args.kwargs["max_workers"], 1)
            self.assertEqual(main(args), 0)
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                main(args + ["--seed", "99"])
            self.assertEqual(main(args + ["--model", "flux-2-dev", "--output", str(self.root / "flux2")]), 0)
            self.assertEqual(editor.call_args.args[:2], ("flux-2-dev", "fixed-revision"))
            self.assertIsNone(editor.call_args.args[2])
            before = judge.detect_refusal.call_count
            with patch.dict("os.environ", {}, clear=True):
                generation_args = args + ["--output", str(self.root / "generation")]
                self.assertEqual(main(generation_args + ["--check"]), 0)
                self.assertEqual(main(generation_args), 0)
            self.assertEqual(judge.detect_refusal.call_count, before)
            self.assertIsNone(benchmark.call_args.kwargs.get("judge"))
            self.assertTrue(benchmark.call_args.kwargs["generation_only"])
        self.assertNotIn("test-secret", (output / "settings.json").read_text())


if __name__ == "__main__":
    unittest.main()
