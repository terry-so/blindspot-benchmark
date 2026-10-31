"""Save and reuse the selected source cases for a CLI run."""
import hashlib
import json
import math
import random
from pathlib import Path
from shutil import copyfile

from utils.benchmark_runner import atomic_json, validate_inputs


def execution_cases(cases, percent=100):
    """Limit this invocation without changing the saved dataset or resume settings."""
    if type(percent) is not int or not 1 <= percent <= 100:
        raise ValueError('run_percent must be an integer from 1 to 100')
    if percent == 100:
        return cases
    groups = {}
    for case in cases:
        groups.setdefault(case['subcategory'], []).append(case)
    rng = random.Random(42)
    selected = []
    for _, group in sorted(groups.items()):
        group = sorted(group, key=lambda c: c['case_id'])
        rng.shuffle(group)
        selected.extend(group[:math.ceil(len(group) * percent / 100)])
    return selected


def fingerprint(cases):
    digest = hashlib.sha256()
    for case in cases:
        metadata = {k: v for k, v in case.items() if k != 'image_path'}
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        digest.update(hashlib.sha256(Path(case['image_path']).read_bytes()).digest())
    return digest.hexdigest()


def run_cases(output, dataset, count=None, smoke=False, save=False):
    from run_local import load_cases

    folder = Path(output) / 'data'
    manifest = folder / 'selection.json'
    selection = {'dataset': str(dataset), 'cases_per_subcategory': count, 'smoke_test': smoke}
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['selection'] != selection:
            raise ValueError('Case selection changed. Use a new --output directory.')
        # Snapshot paths are relative to data/, so the run folder can be moved.
        cases = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
        for case in cases:
            case['image_path'] = str((folder / case['image_path']).resolve())
        if fingerprint(cases) != saved['sha256']:
            raise ValueError('Saved dataset changed. Restore it or use a new output directory.')
        return cases
    cases = load_cases(dataset, count)
    if smoke:
        cases = cases[:1]
    validate_inputs(cases, ['snapshot'], ['L1'], 1, 1, 0)
    if not save:
        return cases
    if folder.exists():
        raise ValueError('Incomplete data snapshot. Use a new output directory.')
    saved_cases = []
    for case in cases:
        destination = folder / case['case_id'] / 'input.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(case['image_path'], destination)
        saved_case = case | {'image_path': f"{case['case_id']}/input.png"}
        atomic_json(destination.with_name('case.json'), saved_case)
        saved_cases.append(saved_case)
    atomic_json(folder / 'dataset.json', saved_cases)
    atomic_json(manifest, {'selection': selection, 'sha256': fingerprint(cases)})
    return run_cases(output, dataset, count, smoke)
