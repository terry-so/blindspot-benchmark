"""Prepare and inspect datasets before running benchmark experiments."""

from utils.exponential_backoff import retry_with_exponential_backoff
import json
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from models.inference import create_edit_prompt


def validate_categories(cases):
    """Check the dataset taxonomy once during preparation."""
    for case in cases:
        domain = case["domain"]
        if domain not in {f"D{i}" for i in range(1, 7)}:
            raise ValueError(f"Invalid domain: {domain}")
        if case["subcategory"] not in {f"{domain[1:]}.{i}" for i in range(1, 6)}:
            raise ValueError(f"Invalid subcategory for {domain}: {case['subcategory']}")


def generate_seed_instance(subcategory_template_path, output_path, model):
    gen_seed = retry_with_exponential_backoff(model.generate_seed_instance)
    instance = gen_seed(subcategory_template_path)
    with open(output_path, "w") as f:
        f.write(instance)

    instance = json.loads(instance)
    return instance


def generate_seed_image(seed_json_path, prompt_path, output_path, model):
    with open(seed_json_path, "r") as f:
        seed = json.load(f)

    with open(prompt_path, "r") as f:
        prompt = f.read()
    scene_spec = seed["scene_spec"]
    edit_spec = seed["edit_spec"]
    prompt = prompt.replace("{scene type}", scene_spec["scene_type"])
    prompt = prompt.replace("{object}", scene_spec["object"])
    prompt = prompt.replace("{location}", scene_spec["location"])
    prompt = prompt.replace("{attribute}", edit_spec["attribute"])
    prompt = prompt.replace("{original_value}", edit_spec["original_value"])
    prompt = prompt.replace("{camera}", scene_spec["appearance"]["camera"])
    prompt = prompt.replace("{lighting}", scene_spec["appearance"]["lighting"])
    gen_img = retry_with_exponential_backoff(model.generate_seed_image)
    return gen_img(prompt, output_path)


def generate_cases(image_model, prompt_model, counts, use_reviewed=True):
    """Create one run with the requested number of cases per subcategory."""
    templates = {}
    for subcategory, count in counts.items():
        if type(count) is not int or count < 0:
            raise ValueError("Case counts must be nonnegative integers")
        if subcategory not in {f"{d}.{s}" for d in range(1, 7) for s in range(1, 6)}:
            raise ValueError(f"Invalid subcategory: {subcategory}")
        parts = subcategory.split(".")

        domain, number = parts
        template = (
            Path("data_foundry/subcategory_template")
            / f"D{domain}"
            / f"D{domain}{number}.yaml"
        )
        if not use_reviewed and count > 0 and not template.is_file():
            raise FileNotFoundError(f"Template not found for {subcategory}: {template}")
        templates[subcategory] = template

    if not any(counts.values()):
        raise ValueError("Request at least one case")

    selected = {}
    if use_reviewed:
        reviewed = json.loads(Path("data/reviewed/dataset.json").read_text())
        for subcategory, count in counts.items():
            matches = [case for case in reviewed if case["subcategory"] == subcategory]
            if len(matches) < count:
                raise ValueError(
                    f"Only {len(matches)} reviewed cases for "
                    f"{subcategory}; requested {count}"
                )
            selected[subcategory] = matches[:count]
            validate_categories(selected[subcategory])
            for case in selected[subcategory]:
                if not Path(case["image_path"]).is_file():
                    raise FileNotFoundError(
                        f"Reviewed image is missing: {case['image_path']}"
                    )

    run_dir = Path("runs") / (f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
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
                    raise ValueError("Generated instance has the wrong subcategory")
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
            validate_categories([case])
            (case_dir / "case.json").write_text(json.dumps(case, indent=2))
            cases.append(case)

    (run_dir / "data" / "dataset.json").write_text(json.dumps(cases, indent=2))
    return cases, run_dir


def inspect_case(case):
    """Show the seed image and the prompts before running the benchmark."""
    from IPython.display import display
    from PIL import Image

    print(case["case_id"], case["subcategory"])
    print("Target:", case["verification"]["expected_result"])
    for level in ("L1", "L2a", "L2b", "L3"):
        print(f"{level}: {case['prompts'].get(level)}")
    with Image.open(case["image_path"]) as image:
        display(image.copy())


def inspect_cases(cases, columns=3, page_size=12, page=1):
    """Show a page of images; use inspect_case for full prompts and details."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from textwrap import fill, shorten

    if any(type(value) is not int or value < 1 for value in (columns, page_size, page)):
        raise ValueError("columns, page_size, and page must be positive integers")
    start = (page - 1) * page_size
    selected = cases[start:start + page_size]
    if not selected:
        print("No cases on this page.")
        return

    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    try:
        for ax in axes.flat:
            ax.axis("off")
        for index, (ax, case) in enumerate(zip(axes.flat, selected), start=start):
            with Image.open(case["image_path"]) as image:
                ax.imshow(image.convert("RGB"))
            target = shorten(case["verification"]["expected_result"], width=100)
            ax.set_title(f"cases[{index}] · {case['subcategory']}\n{fill(target, 40)}", fontsize=9)
        fig.suptitle(f"Page {page} · Cases {start + 1}–{start + len(selected)} of {len(cases)}")
        fig.tight_layout()
        plt.show()
    finally:
        plt.close(fig)



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

    validate_categories(by_id.values())
    dataset_path.write_text(json.dumps(list(by_id.values()), indent=2))
    return dataset_path
