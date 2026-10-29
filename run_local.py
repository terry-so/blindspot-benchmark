"""Run one local editor sequentially through the existing benchmark."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from models.local_editors import LocalEditor, MODEL_IDS, PUP_ID
from run_benchmark import run_benchmark
from utils.benchmark_runner import atomic_json, validate_inputs
from utils.run_dataset import run_cases, fingerprint

ROOT = Path(__file__).resolve().parent


def load_cases(dataset, cases_per_subcategory=None):
    if Path(dataset).is_dir():
        from data_foundry.generate_seed import update_reviewed_dataset
        dataset = update_reviewed_dataset(Path(dataset).resolve())
    cases = json.loads(Path(dataset).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a nonempty JSON list")
    for case in cases:
        path = Path(case["image_path"].replace("\\", "/"))
        if not path.is_absolute():
            candidates = [ROOT / path, Path(dataset).resolve().parent / path]
            path = next((p for p in candidates if p.is_file()), candidates[0])
        case["image_path"] = str(path.resolve())
    if cases_per_subcategory is not None:
        if type(cases_per_subcategory) is not int or cases_per_subcategory < 1:
            raise ValueError('cases_per_subcategory must be a positive integer')
        groups = {}
        for case in cases:
            groups.setdefault(case['subcategory'], []).append(case)
        selected = []
        for subcategory, group in sorted(groups.items()):
            if len(group) < cases_per_subcategory:
                raise ValueError(f'Subcategory {subcategory} has only {len(group)} cases; requested {cases_per_subcategory}')
            selected.extend(sorted(group, key=lambda case: case['case_id'])[:cases_per_subcategory])
        cases = selected
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/benchmark_cases",
                        help="Reviewed case directory or dataset JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge", default=os.environ.get("JUDGE_MODEL"), help="Existing Azure judge deployment")
    parser.add_argument("--levels", nargs="+", default=["L1"], choices=["L0", "L1", "L2a", "L2b", "L3-reset"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases-per-subcategory", type=int, help="First N cases by ID per subcategory; omitted uses all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run the first case, one repeat, with the same judge")
    parser.add_argument("--check", action="store_true", help="Check inputs, CUDA, imports and model access without loading weights")
    parser.add_argument("--skip-judge", action="store_true", help="Save generations without judging or Azure access")
    parser.add_argument("--snapshot-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.skip_judge and (not args.judge or not all(os.environ.get(key) for key in ("AZURE_OPENAI_BASE_URL", "Azure_API_KEY"))):
        parser.error("Set JUDGE_MODEL (or --judge), AZURE_OPENAI_BASE_URL and Azure_API_KEY")
    snapshot_root = args.snapshot_root or args.output
    cases = run_cases(snapshot_root, args.dataset, args.cases_per_subcategory, args.smoke_test)
    if args.smoke_test:
        args.repeats = 1
    validate_inputs(cases, [args.model], args.levels, args.repeats, 1, 0)

    import torch
    from huggingface_hub import HfApi

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting BF16 is required. Check nvidia-smi and your PyTorch installation.")
    if args.model == "qwen-image-2.1":
        from diffusers import QwenImage21Pipeline
    elif args.model == "flux-2-dev":
        from diffusers import Flux2Pipeline
    else:
        from diffusers import ModularPipeline
        from transformers import Qwen3_5ForConditionalGeneration
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    settings_path = args.output / "settings.json"
    previous = json.loads(settings_path.read_text()) if settings_path.exists() else None
    revisions = previous["revisions"] if previous else {
        repo: HfApi().model_info(repo).sha
        for repo in ([MODEL_IDS[args.model]] + ([PUP_ID] if args.model == "flux-3-klein" else []))
    }
    for repo, revision in revisions.items():
        HfApi().auth_check(repo_id=repo, repo_type="model")
    settings = {
        "model": args.model, "judge": None if args.skip_judge else args.judge,
        "judge_endpoint": None if args.skip_judge else os.environ["AZURE_OPENAI_BASE_URL"],
        "dataset_sha256": fingerprint(cases),
        "case_ids": [case["case_id"] for case in cases], "levels": args.levels,
        "repeats": args.repeats, "seed": args.seed, "cpu_offload": args.cpu_offload,
        "revisions": revisions,
        "packages": {name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "openai")},
    }
    if args.skip_judge:
        settings["generation_only"] = True
    if previous and previous != settings:
        raise ValueError("Run settings changed. Use a new --output directory.")
    if args.check:
        print("Preflight passed. Run the smoke test to verify inference." if args.skip_judge else
              "Preflight passed. Judge credentials are present; the smoke test verifies API access and inference.")
        return 0
    if not previous and args.output.exists() and any(p.name != "data" for p in args.output.iterdir()):
        raise ValueError("Output directory must be empty for a new run")
    cases = run_cases(snapshot_root, args.dataset, args.cases_per_subcategory, args.smoke_test, save=True)
    atomic_json(settings_path, settings)
    if not previous:
        (args.output / "environment.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8"
        )
    judge = None
    if not args.skip_judge:
        from models.wrappers import OpenAIJudge
        judge = OpenAIJudge(args.judge)
        # Validate the configured API before spending time loading GPU weights.
        answer = judge.detect_refusal("Reply with exactly COMPLY.")
        if "COMPLY" not in answer:
            raise RuntimeError(f"Judge connectivity check returned an unexpected response: {answer!r}")
    editor = LocalEditor(
        args.model, revisions[MODEL_IDS[args.model]], revisions.get(PUP_ID),
        seed=args.seed, cpu_offload=args.cpu_offload,
    )
    options = {"generation_only": True} if args.skip_judge else {}
    summary = run_benchmark(
        cases, models={args.model: editor}, judge=judge, levels=args.levels,
        repeats=args.repeats, result_dir=args.output / "results", max_workers=1, **options,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] or summary["unavailable"] else 0


if __name__ == "__main__":
    sys.exit(main())
