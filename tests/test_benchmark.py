import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from run_benchmark import run_benchmark
from utils.benchmark_runner import atomic_json, read_json
from utils.exponential_backoff import retry_with_exponential_backoff


class Model:
    model_name = "mock-editor"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def generate_edit(self, prompt, source, target):
        self.calls.append((prompt, source, target))
        if self.fail:
            raise ValueError("INVALID_ARGUMENT")
        image = Image.new("RGB", (32, 32), "red")
        image.save(target)
        return {
            "status": "success",
            "image": image,
            "text_response": "edited",
            "error": None,
        }


class Judge:
    model_name = "mock-judge"

    def __init__(self, fail=False):
        self.fail = fail

    def detect_refusal(self, prompt):
        return "TECHNICAL_ERROR" if "INVALID_ARGUMENT" in prompt else "COMPLY"

    def judge_fidelity(self, *args):
        if self.fail:
            raise ValueError("bad grade")
        return 1

    def judge_realism(self, *args):
        return 4


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.png"
        Image.new("RGB", (32, 32), "white").save(self.source)
        self.case = {
            "case_id": "c1",
            "domain": "D1",
            "subcategory": "1.1",
            "image_path": str(self.source),
            "prompts": {"L1": "edit", "L2b": "literal edit", "L3": ["one", "two", "three"]},
            "verification": {"expected_result": "red"},
        }
        self.result = self.root / "results" / "model" / "c1_L1_0.json"
        self.distance = patch(
            "evaluators.auto_grader.image_distance",
            side_effect=AssertionError("Refusal scoring must not use LPIPS"),
        )
        self.distance.start()
        self.addCleanup(self.distance.stop)

    def run_batch(self, model=None, judge=None, **kwargs):
        return run_benchmark(
            kwargs.pop("cases", [self.case]),
            {"model": model or Model()},
            judge or Judge(),
            0.01,
            result_dir=self.root / "results",
            progress=False,
            **kwargs
        )

    def test_l0_fidelity_uses_control_prompt_only(self):
        control = "Increase brightness while preserving all text."
        self.case["prompts"].update(L0=control, L2a="pretext", L2b="literal")
        with patch("evaluators.auto_grader.check_fidelity", return_value=1) as fidelity:
            summary = self.run_batch(levels=["L0", "L1", "L2a", "L2b", "L3-reset"])
        self.assertEqual(summary["successful"], 5)
        self.assertEqual(
            [call.args[3] for call in fidelity.call_args_list],
            [control] + [self.case["prompts"]["L2b"]] * 4,
        )

    def test_success_compatible_schema_and_resume(self):
        model = Model()
        self.run_batch(model)
        before = self.result.read_bytes()
        summary = self.run_batch(model)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(before, self.result.read_bytes())
        self.assertEqual(
            set(read_json(self.result)),
            {
                "model",
                "case_id",
                "domain",
                "subcategory",
                "level",
                "repeat",
                "output_path",
                "estimated_cost_usd",
                "generation_pricing",
                "execution_status",
                "text_response",
                "execution_error",
                "block_reason",
                "finish_reason",
                "safety_block",
                "status",
                "r",
                "e",
                "q",
            },
        )

    def test_grading_retry_does_not_generate(self):
        model = Model()
        self.run_batch(model, Judge(True))
        self.assertEqual(len(model.calls), 1)
        self.run_batch(model)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(read_json(self.result)["status"], "ok")
        self.assertEqual(read_json(self.result)["text_response"], "edited")

    def test_generation_failure_retries(self):
        model = Model(True)
        self.run_batch(model)
        model.fail = False
        self.run_batch(model)
        self.assertEqual(len(model.calls), 2)

    def test_corrupt_png_regenerates(self):
        model = Model()
        self.run_batch(model, Judge(True))
        self.result.with_suffix(".png").write_bytes(b"bad")
        self.run_batch(model)
        self.assertEqual(len(model.calls), 2)

    def test_regrade_only_no_editor_even_factory(self):
        run_benchmark(
            [self.case],
            model_factories={"model": lambda: self.fail("editor created")},
            judge_factory=Judge,
            regrade_only=True,
            result_dir=self.root / "results",
            progress=False,
        )
        self.assertFalse(self.result.exists())

    def test_l3_chain_and_final_regrade(self):
        model = Model()
        self.run_batch(model, Judge(True), levels=["L3-reset"])
        self.assertEqual(len(model.calls), 3)
        for i in range(1, 3):
            self.assertEqual(model.calls[i][1], model.calls[i - 1][2])
        self.run_batch(model, levels=["L3-reset"])
        self.assertEqual(len(model.calls), 3)
        result = read_json(self.result.with_name("c1_L3-reset_0.json"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["steps"]), 3)

    def test_l3_interrupted_restarts(self):
        model = Model()
        original = model.generate_edit

        def edit(prompt, *args):
            if prompt == "two":
                raise ValueError("INVALID_ARGUMENT")
            return original(prompt, *args)

        model.generate_edit = edit
        self.run_batch(model, levels=["L3-reset"])
        self.assertFalse(self.result.with_name("c1_L3-reset_0.png").exists())
        model.generate_edit = original
        self.run_batch(model, levels=["L3-reset"])
        self.assertEqual([c[0] for c in model.calls], ["one", "one", "two", "three"])

    def test_provider_refusal(self):
        model = Model()
        model.generate_edit = lambda *a: {
            "status": "failed",
            "image": None,
            "safety_block": True,
            "block_reason": "SAFETY",
        }
        summary = self.run_batch(model)
        self.assertEqual(summary["refused"], 1)
        self.assertEqual(read_json(self.result)["r"], 1)

    def test_parallel_models_and_repeats(self):
        barrier = threading.Barrier(2)

        class Parallel(Model):
            def generate_edit(self, *args):
                barrier.wait(timeout=5)
                return super().generate_edit(*args)

        result = run_benchmark(
            [self.case],
            thresholds=0.01,
            model_factories={"a": Parallel, "b": Parallel},
            judge_factory=Judge,
            levels=["L1", "L1"],
            repeats=2,
            max_workers=4,
            result_dir=self.root / "results",
            progress=False,
        )
        self.assertEqual(result["successful"], 4)

    def test_atomic_write_failure_preserves_previous(self):
        atomic_json(self.root / "atomic.json", {"status": "ok"})
        with self.assertRaises(ValueError):
            atomic_json(self.root / "atomic.json", {"x": float("nan")})
        self.assertEqual(read_json(self.root / "atomic.json"), {"status": "ok"})
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_permanent_not_retried_transient_retried(self):
        class Error(Exception):
            status_code = 400

        attempts = []

        def fail():
            attempts.append(1)
            raise Error()

        with self.assertRaises(Error):
            retry_with_exponential_backoff(fail, max_attempts=3, initial_sleep_time=0)()
        self.assertEqual(len(attempts), 1)
        Error.status_code = 429
        attempts.clear()
        with self.assertRaises(Error):
            retry_with_exponential_backoff(fail, max_attempts=3, initial_sleep_time=0)()
        self.assertEqual(len(attempts), 3)

    def test_legacy_grading_failure_reuses_image_without_checkpoint(self):
        model = Model()
        self.run_batch(model, Judge(True))
        for p in (self.root / "results_run_metadata" / "checkpoints").rglob("*.json"):
            p.unlink()
        self.run_batch(model, regrade_only=True)
        self.assertEqual(len(model.calls), 1)

    def test_missing_png_legacy_success_never_overwritten(self):
        self.result.parent.mkdir(parents=True)
        atomic_json(self.result, {"status": "ok", "r": 1, "e": None, "q": None})
        model = Model()
        result = self.run_batch(model)
        self.assertEqual(result["skipped"], 1)
        self.assertFalse(model.calls)


if __name__ == "__main__":
    unittest.main()
