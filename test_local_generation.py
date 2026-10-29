import argparse
import json
from pathlib import Path
from shutil import copyfile

import torch
from huggingface_hub import HfApi

from models.local_editors import LocalEditor, MODEL_IDS, PUP_ID
from run_local import load_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--dataset", default="data/benchmark_cases")
    parser.add_argument("--output", default="runs/test-local-generation")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting BF16 is required.")

    # Use the same first case for every model.
    cases = load_cases(args.dataset)
    case = sorted(cases, key=lambda item: item["case_id"])[0]
    prompt = case["prompts"]["L1"]

    folder = Path(args.output) / args.model
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / "edited.png"
    output_path.unlink(missing_ok=True)

    copyfile(case["image_path"], folder / "original.png")
    (folder / "case.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    api = HfApi()
    revision = api.model_info(MODEL_IDS[args.model]).sha
    pup_revision = (
        api.model_info(PUP_ID).sha
        if args.model == "flux-3-klein"
        else None
    )

    print(f"Loading {args.model}", flush=True)
    editor = LocalEditor(
        args.model,
        revision=revision,
        pup_revision=pup_revision,
        seed=42,
        cpu_offload=True,
    )

    print(f"Generating {case['case_id']} at L1", flush=True)
    result = editor.generate_edit(
        prompt, case["image_path"], str(output_path)
    )

    if result["status"] != "success" or not output_path.is_file():
        raise RuntimeError(result.get("error") or "No image was saved.")

    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()