import json
from pathlib import Path

from data_foundry.generate_seed import generate_seed_image
from models.inference import create_edit_prompt


def build_dataset(image_model, prompt_model):
    cases = []
    seed_dir = Path("data/seeds")
    seed_dir.mkdir(parents=True, exist_ok=True)

    for domain in ("D3", "D4"):
        root = Path("data_foundry/edit_instance") / domain

        for spec_path in sorted(root.glob("*/*.json")):
            instance = json.loads(spec_path.read_text())
            case_id = f"{domain}_{spec_path.parent.name}_{spec_path.stem}"

            image_path = seed_dir / f"{case_id}.png"
            prompts_path = seed_dir / f"{case_id}_prompts.json"

            if not image_path.exists():
                generate_seed_image(
                    spec_path,
                    "data_foundry/prompts/seed_image_gen_prompt.txt",
                    image_path,
                    model=image_model,
                )
                if not image_path.exists():
                    raise RuntimeError(f"No seed image generated for {case_id}")

            if not prompts_path.exists():
                raw = create_edit_prompt(
                    prompt_model,
                      r"data_foundry/prompts/prompt_gen_prompt.txt",
                      instance)
                
                prompts = {
                    "L0": raw["L0"],
                    "L1": raw["L1"],
                    "L2a": raw["L2A"],
                    "L2b": raw["L2B"],
                    "L3": raw["L3"],
                }
                prompts_path.write_text(json.dumps(prompts, indent=2))

            cases.append({
                "case_id": case_id,
                "domain": domain,
                "subcategory": instance["subcategory_id"],
                "image_path": str(image_path),
                "instance": instance,
                "verification": instance["verification"],
                "prompts": json.loads(prompts_path.read_text()),
            })

    Path("data/dataset.json").write_text(json.dumps(cases, indent=2))

