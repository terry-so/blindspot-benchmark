import json
from pathlib import Path

from models.inference import execute_edit
from evaluators.auto_grader import grade_output


def run_case(model, judge, case, level, output_path, threshold):
    instruction = case["prompts"][level]

    output = execute_edit(
        model,
        instruction,
        case["image_path"],
        str(output_path),
    )

    return grade_output(judge, case, instruction, output, str(output_path), threshold)


def run_benchmark(cases, models, judge, thresholds, levels=("L1",), repeats=1):
    for model_name, model in models.items():
        result_dir = Path("results") / model_name
        result_dir.mkdir(parents=True, exist_ok=True)

        for case in cases:
            for level in levels:
                for repeat in range(repeats):
                    stem = f"{case['case_id']}_{level}_{repeat}"
                    result_path = result_dir / f"{stem}.json"
                    output_path = result_dir / f"{stem}.png"

                    if result_path.exists():
                        saved = json.loads(result_path.read_text())
                        if saved["status"] == "ok":
                            continue

                    metadata = {
                        "model": model_name,
                        "case_id": case["case_id"],
                        "domain": case["domain"],
                        "subcategory": case["subcategory"],
                        "level": level,
                        "repeat": repeat,
                        "output_path": str(output_path),
                    }

                    try:
                        labels = run_case(
                            model, judge, case, level, output_path,
                            thresholds[model_name],
                        )
                    except Exception as e:
                        labels = {
                            "status": "error",
                            "r": None,
                            "e": None,
                            "q": None,
                            "error": str(e),
                        }

                    result_path.write_text(
                        json.dumps(metadata | labels, indent=2)
                    )


def run_chain(
    model, judge, case, protocol, output_dir, threshold
):
    """
    Return one result for the entire chain, including:
      r, e, q, first_refusal_step, chain_length

    1. Start with the original seed image.
    2. For each step:
       - session: retain the conversation history
       - reset: fresh context, previous output image only
       - execute the step
       - detect refusal against that step's input image
       - save output and response metadata
       - feed output image into the next step
    3. Grade final fidelity against the ORIGINAL case target.
    4. Grade final realism.
    """
    ...