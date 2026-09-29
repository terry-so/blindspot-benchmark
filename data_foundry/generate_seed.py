from models.wrappers import GeminiModel
from utils.exponential_backoff import retry_with_exponential_backoff
import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from models.inference import create_edit_prompt

def generate_seed_instance(subcategory_template_path, output_path, model):
    gen_seed = retry_with_exponential_backoff(model.generate_seed_instance)
    instance = gen_seed(subcategory_template_path)
    with open(output_path, 'w') as f:
        f.write(instance)

    instance = json.loads(instance)
    return instance

def generate_seed_image(seed_json_path, prompt_path, output_path, model):
    with open(seed_json_path, 'r') as f:
        seed = json.load(f)

    with open(prompt_path,'r') as f:
        prompt = f.read()
    scene_spec = seed['scene_spec']
    edit_spec = seed['edit_spec']
    prompt = prompt.replace('{scene type}',scene_spec['scene_type'])
    prompt = prompt.replace('{object}',scene_spec['object'])
    prompt = prompt.replace('{location}',scene_spec['location'])
    prompt = prompt.replace('{attribute}',edit_spec['attribute'])
    prompt = prompt.replace('{original_value}',edit_spec['original_value'])
    prompt = prompt.replace('{camera}',scene_spec['appearance']['camera'])
    prompt = prompt.replace('{lighting}',scene_spec['appearance']['lighting'])
    gen_img = retry_with_exponential_backoff(model.generate_seed_image)
    return gen_img(prompt, output_path)


def generate_cases(image_model, prompt_model, counts, use_reviewed=True):
    """Create one run with the requested number of cases per subcategory."""
    templates = {}
    for subcategory, count in counts.items():
        parts = subcategory.split(".")

        domain, number = parts
        template = (
            Path("data_foundry/subcategory_template")
            / f"D{domain}"
            / f"D{domain}{number}.yaml"
        )
        if not template.is_file():
            raise FileNotFoundError(
                f"Template not found for {subcategory}: {template}"
            )
        templates[subcategory] = template

    if not any(counts.values()):
        raise ValueError("Request at least one case")

    selected = {}
    if use_reviewed:
        reviewed = json.loads(Path("data/reviewed/dataset.json").read_text())
        for subcategory, count in counts.items():
            matches = [
                case for case in reviewed
                if case["subcategory"] == subcategory
            ]
            if len(matches) < count:
                raise ValueError(
                    f"Only {len(matches)} reviewed cases for "
                    f"{subcategory}; requested {count}"
                )
            selected[subcategory] = matches[:count]
            for case in selected[subcategory]:
                if not Path(case["image_path"]).is_file():
                    raise FileNotFoundError(
                        f"Reviewed image is missing: {case['image_path']}"
                    )

    run_dir = Path("runs") / (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    cases = []
    for subcategory, count in counts.items():
        for index in range(count):
            case_id = (
                selected[subcategory][index]["case_id"]
                if use_reviewed
                else f"D{subcategory.replace('.', '_')}_{index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            case_dir = run_dir / "data" / case_id
            case_dir.mkdir(parents=True)
            image_path = case_dir / "input.png"

            if use_reviewed:
                case = selected[subcategory][index].copy()
                copyfile(case["image_path"], image_path)
            else:
                instance_path = case_dir / "instance.json"
                instance = generate_seed_instance(
                    templates[subcategory], instance_path, prompt_model
                )
                if instance["subcategory_id"] != subcategory:
                    raise ValueError(
                        "Generated instance has the wrong subcategory"
                    )
                generate_seed_image(
                    instance_path,
                    "data_foundry/prompts/seed_image_gen_prompt.txt",
                    image_path,
                    image_model,
                )
                raw = create_edit_prompt(
                    prompt_model,
                    "data_foundry/prompts/prompt_gen_prompt.txt",
                    instance,
                )
                case = {
                    "case_id": case_id,
                    "domain": f"D{subcategory.split('.')[0]}",
                    "subcategory": subcategory,
                    "instance": instance,
                    "verification": instance["verification"],
                    "prompts": {
                        "L0": raw["L0"],
                        "L1": raw["L1"],
                        "L2a": raw["L2A"],
                        "L2b": raw["L2B"],
                        "L3": raw["L3"],
                    },
                }

            case["image_path"] = str(image_path)
            (case_dir / "case.json").write_text(
                json.dumps(case, indent=2)
            )
            cases.append(case)

    (run_dir / "data" / "dataset.json").write_text(
        json.dumps(cases, indent=2)
    )
    return cases, run_dir


def inspect_case(case):
    """Show the seed image and the prompts before running the benchmark."""
    from IPython.display import display
    from PIL import Image

    print(case["case_id"], case["subcategory"])
    print("Target:", case["verification"]["expected_result"])
    for level in ("L1", "L2a", "L2b",'L3'):
        print(f"{level}: {case['prompts'].get(level)}")
    with Image.open(case["image_path"]) as image:
        display(image.copy())


def inspect_cases(cases):
    for case in cases:
        inspect_case(case)
        
        


def update_reviewed_dataset(reviewed_dir="data/reviewed"):
    """Add copied review-case folders to the reviewed dataset."""
    reviewed_dir = Path(reviewed_dir)
    dataset_path = reviewed_dir / "dataset.json"
    cases = json.loads(dataset_path.read_text()) if dataset_path.exists() else []
    by_id = {case["case_id"]: case for case in cases}

    for case_path in sorted(reviewed_dir.glob("*/case.json")):
        case = json.loads(case_path.read_text())
        if case["case_id"] != case_path.parent.name:
            raise ValueError(f"Case ID does not match folder: {case_path}")

        image_path = case_path.parent / "input.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Reviewed image is missing: {image_path}")

        case["image_path"] = str(image_path)
        by_id[case["case_id"]] = case

    dataset_path.write_text(json.dumps(list(by_id.values()), indent=2))
    return dataset_path









if __name__ == "__main__":
    pass
