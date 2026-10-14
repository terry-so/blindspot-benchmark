"""Small file and recovery helpers used by run_benchmark.py."""

import json
import math
import os
import tempfile
from pathlib import Path

from PIL import Image


def atomic_json(path, value):
    """Replace the JSON only after its complete replacement has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def validate_inputs(cases, names, levels, repeats, max_workers, threshold):
    """Fail before any API calls if inputs could collide or cannot be evaluated."""
    for group in (names, [case["case_id"] for case in cases]):
        for name in group:
            if (
                not isinstance(name, str)
                or not name.strip()
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
            ):
                raise ValueError(f"Names must be nonempty path components: {name!r}")
        if len(group) != len(set(group)):
            raise ValueError("Duplicate model names or case IDs")
    if (
        not names
        or not levels
        or any(level not in {"L0", "L1", "L2a", "L2b", "L3-reset"} for level in levels)
    ):
        raise ValueError("Provide at least one model and supported level")
    for value in (repeats, max_workers):
        if type(value) is not int or value < 1:
            raise ValueError("repeats and max_workers must be positive integers")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("thresholds must be a finite, nonnegative number")
    for case in cases:
        if not valid_image(case["image_path"]):
            raise ValueError(f'Unreadable source image: {case["image_path"]}')
        target = case["verification"]["expected_result"]
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Missing expected_result")
        for level in levels:
            prompts = (
                case["prompts"]["L3"]
                if level == "L3-reset"
                else [case["prompts"][level]]
            )
            if (
                not isinstance(prompts, list)
                or not prompts
                or any(not isinstance(p, str) or not p.strip() for p in prompts)
            ):
                raise ValueError(f"Invalid prompts for {level}")


def can_regrade(saved, image_path, level):
    """A PNG alone is not proof that generation completed successfully."""
    if saved.get("execution_status") != "success" or not valid_image(image_path):
        return False
    if level != "L3-reset":
        return True
    steps = saved.get("steps")
    return (
        isinstance(steps, list)
        and bool(steps)
        and len(steps) == saved.get("chain_length")
        and saved.get("failed_step") is None
        and saved.get("first_refusal_step") is None
        and all(isinstance(step, dict) and step.get("r") == 0 for step in steps)
    )


def load_generation(saved, image_path):
    """Restore an editor response and metadata without calling a model or judge."""
    response = {
        key: saved.get(key)
        for key in (
            "execution_status",
            "text_response",
            "execution_error",
            "block_reason",
            "finish_reason",
            "safety_block",
        )
    }
    for key in ("chain_length", "steps", "first_refusal_step", "failed_step"):
        if key in saved:
            response[key] = saved[key]
    with Image.open(image_path) as image:
        output = {
            "status": "success",
            "image": image.copy(),
            "error": saved.get("execution_error"),
            "text_response": saved.get("text_response"),
            "block_reason": saved.get("block_reason"),
            "finish_reason": saved.get("finish_reason"),
            "safety_block": saved.get("safety_block", False),
        }
    return output, response


def response_metadata(output):
    """Fields shared by single edits, L3 steps, and generation checkpoints."""
    return {
        "execution_status": output["status"],
        "text_response": output.get("text_response"),
        "execution_error": output.get("error"),
        "block_reason": output.get("block_reason"),
        "finish_reason": output.get("finish_reason"),
        "safety_block": output.get("safety_block", False),
    }
