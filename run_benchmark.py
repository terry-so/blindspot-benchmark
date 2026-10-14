"""Run independent experiments in parallel; save results in the existing format."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluators.auto_grader import grade_output
from models.inference import generate_single_edit, generate_l3_chain
from utils.benchmark_runner import (
    atomic_json,
    read_json,
    validate_inputs,
    can_regrade,
    load_generation,
)


def run_experiment(
    case,
    name,
    level,
    repeat,
    *,
    make_model,
    make_judge,
    result_dir,
    threshold,
    regrade_only=False,
):
    """Recover or generate one output, then grade it and save the result."""
    result_path = Path(result_dir) / name / f"{case['case_id']}_{level}_{repeat}.json"
    output_path = result_path.with_suffix(".png")
    saved = read_json(result_path)
    if saved.get("status") == "ok":
        return "skipped"

    recover = can_regrade(saved, output_path, level)
    if regrade_only and not recover:
        return "unavailable"

    metadata = {
        "model": name,
        "case_id": case["case_id"],
        "domain": case["domain"],
        "subcategory": case["subcategory"],
        "level": level,
        "repeat": repeat,
        "output_path": str(output_path),
    }
    # Keep the old pending/error representation so historical readers still work.
    pending = {
        "status": "error",
        "r": None,
        "e": None,
        "q": None,
        "error": "Grading not completed",
    }
    if not recover:
        # Invalidate old generation metadata before replacing its image.
        atomic_json(result_path, metadata | pending)
        output_path.unlink(missing_ok=True)

    refusal_score = 0 if recover and level == "L3-reset" else None
    try:
        judge = make_judge()
        if recover:
            output, response = load_generation(saved, output_path)
        elif level == "L3-reset":
            output, response = generate_l3_chain(
                make_model(), judge, case, output_path, threshold
            )
        else:
            output, response = generate_single_edit(
                make_model(), case, level, output_path
            )

        # A refusal or failed L3 step ends the experiment before final grading.
        if "status" in response:
            result = response
        else:
            if level == "L3-reset":
                refusal_score = 0  # Every step already passed refusal detection.
            atomic_json(result_path, metadata | response | pending)
            instruction = (
                "\n".join(case["prompts"]["L3"])
                if level == "L3-reset"
                else case["prompts"][level]
            )
            labels = grade_output(
                judge,
                case,
                instruction,
                output,
                str(output_path),
                threshold,
                refusal_score=refusal_score,
            )
            result = response | labels
    except Exception as exc:
        # A completed generation remains available if construction/grading fails.
        result = read_json(result_path)
        result.update(
            status="error",
            r=refusal_score,
            e=None,
            q=None,
            error=str(exc),
        )

    atomic_json(result_path, metadata | result)
    if result["status"] != "ok":
        return "failed"
    return "refused" if result["r"] == 1 else "successful"


def run_benchmark(
    cases,
    models=None,
    judge=None,
    thresholds=0.01,
    levels=("L1",),
    repeats=1,
    result_dir="results",
    *,
    max_workers=1,
    model_factories=None,
    judge_factory=None,
    regrade_only=False,
    progress=True,
):
    """Validate, schedule experiments, and count outcomes.

    API runs use factories returning fresh objects. Local GPU runs may pass
    existing models/judge with max_workers=1. Only one runner may use a results
    directory at a time. The legacy 'successful' count means non-refused;
    fidelity success is recorded separately in each result's e score.
    """
    models = models or {}
    model_factories = model_factories or {}
    if models and model_factories:
        raise ValueError("Use models OR model_factories, not both")
    if judge is not None and judge_factory is not None:
        raise ValueError("Use judge OR judge_factory, not both")
    if judge is None and judge_factory is None:
        raise ValueError("Provide a judge or judge_factory")
    if max_workers > 1 and (models or judge_factory is None):
        raise ValueError("Parallel runs require model_factories and judge_factory")

    # Normalize the sequential object interface once, at the entry point.
    make_models = model_factories or {
        name: (lambda model=model: model) for name, model in models.items()
    }
    make_judge = judge_factory if judge_factory is not None else lambda: judge
    cases = list(cases)
    levels = tuple(dict.fromkeys(levels))
    validate_inputs(cases, list(make_models), levels, repeats, max_workers, thresholds)

    jobs = [
        (case, name, level, repeat)
        for case in cases
        for name in make_models
        for level in levels
        for repeat in range(repeats)
    ]
    summary = dict(
        total=len(jobs), successful=0, refused=0, failed=0, skipped=0, unavailable=0
    )
    # Each job owns its files; an L3 chain runs sequentially inside the job.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                run_experiment,
                case,
                name,
                level,
                repeat,
                make_model=make_models[name],
                make_judge=make_judge,
                result_dir=result_dir,
                threshold=thresholds,
                regrade_only=regrade_only,
            )
            for case, name, level, repeat in jobs
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            summary[future.result()] += 1
            if progress:
                print(f"{completed}/{len(jobs)} finished", flush=True)
    return summary
