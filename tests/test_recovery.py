"""Recovery and concurrency checks using fake editors and judges only."""

import threading
import time
import unittest
from unittest.mock import patch

import test_benchmark as fixtures
from test_benchmark import Model, Judge
from run_benchmark import run_benchmark
from models.inference import generate_l3_chain
from evaluators.auto_grader import detect_refusal
from utils.benchmark_runner import atomic_json, read_json


class RecoveryTests(unittest.TestCase):
    setUp = fixtures.BenchmarkTests.setUp
    run_batch = fixtures.BenchmarkTests.run_batch

    def test_l3_generation_does_not_grade_the_final_image(self):
        model = Model()
        judge = Judge()
        judge.judge_fidelity = lambda *args: self.fail("generation must not grade")
        judge.judge_realism = lambda *args: self.fail("generation must not grade")
        output, metadata = generate_l3_chain(
            model, judge, self.case, self.root / "final.png", 0.01
        )
        self.assertEqual(output["status"], "success")
        self.assertNotIn("status", metadata)
        self.assertEqual([step["r"] for step in metadata["steps"]], [0, 0, 0])
        self.assertTrue((self.root / "final.png").is_file())

    def test_l3_refusal_judge_error_keeps_step_response(self):
        judge = Judge()
        judge.detect_refusal = lambda *args: "INVALID_LABEL"
        summary = self.run_batch(judge=judge, levels=["L3-reset"])
        saved = read_json(self.result.with_name("c1_L3-reset_0.json"))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(saved["failed_step"], 1)
        self.assertEqual(saved["text_response"], "edited")
        self.assertEqual(saved["steps"][0]["execution_status"], "success")
        self.assertFalse(self.result.with_name("c1_L3-reset_0.png").exists())

    def test_interruption_after_generation_regrades_without_editor(self):
        class InterruptedJudge(Judge):
            def judge_fidelity(self, *args):
                raise KeyboardInterrupt("simulated interruption")

        model = Model()
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(model, InterruptedJudge())
        self.assertEqual(read_json(self.result)["execution_status"], "success")
        summary = run_benchmark(
            [self.case],
            model_factories={"model": lambda: self.fail("editor created")},
            judge_factory=Judge,
            regrade_only=True,
            result_dir=self.root / "results",
            progress=False,
        )
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(len(model.calls), 1)

    def test_corrupt_json_requires_regeneration(self):
        model = Model()
        self.run_batch(model, Judge(True))
        self.result.write_text("{")
        self.run_batch(model)
        self.assertEqual(len(model.calls), 2)

    def test_l3_interruption_after_chain_regrades_without_generating(self):
        class InterruptedJudge(Judge):
            def judge_fidelity(self, *args):
                raise KeyboardInterrupt("simulated interruption")

        model = Model()
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(model, InterruptedJudge(), levels=["L3-reset"])
        summary = self.run_batch(model, levels=["L3-reset"], regrade_only=True)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(len(model.calls), 3)

    def test_png_without_metadata_is_not_regraded(self):
        self.result.parent.mkdir(parents=True)
        Model().generate_edit("edit", self.source, self.result.with_suffix(".png"))
        summary = self.run_batch(regrade_only=True)
        self.assertEqual(summary["unavailable"], 1)
        self.assertFalse(self.result.exists())

    def test_parallel_limit_and_fresh_factories(self):
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        active = 0
        peak = 0
        editors = []
        judges = []

        class ParallelModel(Model):
            def generate_edit(self, *args):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    barrier.wait(timeout=5)
                    time.sleep(0.01)
                    return super().generate_edit(*args)
                finally:
                    with lock:
                        active -= 1

        def editor_factory():
            model = ParallelModel()
            editors.append(model)
            return model

        def judge_factory():
            judge = Judge()
            judges.append(judge)
            return judge

        summary = run_benchmark(
            [self.case],
            model_factories={"model": editor_factory},
            judge_factory=judge_factory,
            max_workers=2,
            repeats=6,
            result_dir=self.root / "results",
            progress=False,
        )
        self.assertEqual(summary["successful"], 6)
        self.assertEqual(peak, 2)
        self.assertEqual(len(editors), 6)
        self.assertEqual(len(judges), 6)
        self.assertTrue(all(len(editor.calls) == 1 for editor in editors))

    def test_shared_objects_rejected_in_parallel(self):
        with self.assertRaises(ValueError):
            self.run_batch(max_workers=2)
        self.assertFalse(self.result.exists())

    def test_duplicate_cases_fail_before_generation(self):
        model = Model()
        with self.assertRaises(ValueError):
            self.run_batch(model, cases=[self.case, self.case])
        self.assertFalse(model.calls)

    def test_factory_failure_preserves_recovery(self):
        model = Model()
        self.run_batch(model, Judge(True))

        def broken_judge():
            raise ValueError("bad configuration")

        run_benchmark(
            [self.case],
            models={"model": model},
            judge_factory=broken_judge,
            result_dir=self.root / "results",
            progress=False,
        )
        self.assertEqual(read_json(self.result)["execution_status"], "success")
        self.run_batch(model)
        self.assertEqual(len(model.calls), 1)

    def test_atomic_replace_failure_preserves_previous_file(self):
        atomic_json(self.result, {"status": "ok"})
        with patch(
            "utils.benchmark_runner.os.replace", side_effect=OSError("disk error")
        ):
            with self.assertRaises(OSError):
                atomic_json(self.result, {"status": "error"})
        self.assertEqual(read_json(self.result), {"status": "ok"})
        self.assertFalse(list(self.result.parent.glob("*.tmp")))

    def test_l3_refusal_stops_chain(self):
        model = Model()
        original = model.generate_edit

        def edit(prompt, *args):
            if prompt == "two":
                return {"status": "failed", "image": None, "safety_block": True}
            return original(prompt, *args)

        model.generate_edit = edit
        summary = self.run_batch(model, levels=["L3-reset"])
        result = read_json(self.result.with_name("c1_L3-reset_0.json"))
        self.assertEqual(summary["refused"], 1)
        self.assertEqual(result["first_refusal_step"], 2)
        self.assertEqual([call[0] for call in model.calls], ["one"])
        self.assertFalse(self.result.with_name("c1_L3-reset_0.png").exists())

    def test_unchanged_image_is_not_refusal_and_fidelity_zero(self):
        # The fake editor returns red, so make its output identical to the input.
        fixtures.Image.new("RGB", (32, 32), "red").save(self.source)
        self.run_batch()
        self.assertEqual(read_json(self.result)["r"], 0)
        self.result.unlink()
        judge = Judge()
        judge.judge_fidelity = lambda *args: 0
        judge.judge_realism = lambda *args: self.fail("realism should be skipped")
        self.run_batch(judge=judge)
        self.assertEqual(read_json(self.result)["e"], 0)
        self.assertIsNone(read_json(self.result)["q"])

    def test_refusal_uses_judge_without_image_distance(self):
        for label, expected in (("COMPLY", 0), ("REFUSE", 1)):
            for threshold in (0, 1):
                with self.subTest(label=label, threshold=threshold):
                    judge = Judge()
                    judge.detect_refusal = lambda prompt: label
                    output = {"image": object(), "text_response": "model response"}
                    self.assertEqual(
                        detect_refusal(judge, "unused.png", output, threshold, "edit"),
                        expected,
                    )
        self.assertEqual(
            detect_refusal(Judge(), "unused.png", {"image": object()}, 1, "edit"),
            0,
        )
