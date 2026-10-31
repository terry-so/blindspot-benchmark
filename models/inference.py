"""Execute image edits and L3 chains; final grading belongs to the runner."""

import json
from pathlib import Path
from shutil import copyfile
from typing import Any

from evaluators.auto_grader import detect_refusal
from utils.benchmark_runner import response_metadata
from utils.exponential_backoff import retry_with_exponential_backoff, error_kind


def create_edit_prompt(model, prompt_path: str, seed_instance: dict) -> dict:
    """Generate the level prompts for a seed instance and parse the JSON response."""
    with open(prompt_path, "r") as f:
        prompt = f.read()

    scene_spec = seed_instance["scene_spec"]
    edit_spec = seed_instance["edit_spec"]

    prompt = prompt.replace("{scene_spec}", json.dumps(scene_spec, indent=2))
    prompt = prompt.replace("{edit_spec}", json.dumps(edit_spec, indent=2))

    gen_prompt = retry_with_exponential_backoff(model.generate_edit_prompt)
    return json.loads(gen_prompt(prompt))


def execute_edit(
    model, prompt: str, input_image_path: str, output_image_path: str
) -> dict[str, Any]:
    """Call the editor with retries, returning failures for refusal classification."""
    gen_edit = retry_with_exponential_backoff(model.generate_edit)

    try:
        return gen_edit(prompt, input_image_path, output_image_path)
    except Exception as e:
        return {
            "status": "failed",
            "text_response": None,
            "image": None,
            "error": f"{error_kind(e)}: {e}",
        }


def generate_single_edit(model, case, level, output_path):
    """Return the editor output and serializable metadata for one edit."""
    output = execute_edit(
        model, case["prompts"][level], case["image_path"], str(output_path)
    )
    return output, response_metadata(output)


def generate_l3_chain(model, judge, case, output_path, threshold):
    """Edit the previous image at each step; stop on refusal or technical failure.

    Return (editor_output, metadata). Terminal refusals/errors include a status
    in metadata. A completed chain is ready for the runner to checkpoint and grade.
    """
    prompts = case["prompts"]["L3"]
    output_path = Path(output_path)
    step_dir = output_path.parent / f"{output_path.stem}_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    current_image = str(case["image_path"])
    steps = []
    result = {
        "chain_length": len(prompts),
        "steps": steps,
        "first_refusal_step": None,
        "failed_step": None,
        "safety_block": False,
    }
    for index, instruction in enumerate(prompts, start=1):
        step_path = step_dir / f"step_{index}.png"
        step_path.unlink(missing_ok=True)
        step = {
            "step": index,
            "instruction": instruction,
            "input_path": current_image,
            "output_path": None,
        }
        steps.append(step)
        try:
            output = execute_edit(model, instruction, current_image, str(step_path))
            response = response_metadata(output)
            step.update(response)
            result.update(response)
            if step_path.is_file():
                step["output_path"] = str(step_path)

            refusal = detect_refusal(
                judge,
                current_image,
                output,
                threshold,
                instruction=instruction,
            )
            step["r"] = refusal
            if refusal == 1:
                return output, result | {
                    "status": "ok",
                    "r": 1,
                    "e": None,
                    "q": None,
                    "first_refusal_step": index,
                }
            if refusal != 0:
                raise ValueError(f"Invalid refusal score: {refusal}")
            if output.get("image") is None or not step_path.is_file():
                raise RuntimeError("Step returned no saved image")
            current_image = str(step_path)
        except Exception as exc:
            step["error"] = str(exc)
            return None, result | {
                "status": "error",
                "r": None,
                "e": None,
                "q": None,
                "error": str(exc),
                "failed_step": index,
            }

    # Only complete chains receive a final image and can recover final grading.
    copyfile(current_image, output_path)
    return output, result
