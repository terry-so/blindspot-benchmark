# BLINDSPOT

An image-editing benchmark investigating when models refuse an editing request,
whether they achieve the requested change, and how realistic their outputs are.

> **Work in progress:** 
> Results are preliminary.

## What it does

BLINDSPOT evaluates models using individual editing requests and sequences of
edits, where each step uses the previous step’s output.

Each experiment records:

- **Refusal:** whether refusal was detected.
- **Edit fidelity:** whether the expected change was achieved.
- **Realism:** how realistic the output looks, assessed when fidelity succeeds.

The pipeline supports parallel experiments, saves intermediate results, and
resumes unfinished runs.

## Workflow

1. Load reviewed cases or prepare new cases.
2. Inspect the images, prompts, and expected edits.
3. Run the selected models and prompt levels.
4. Evaluate and save the outputs.
5. Generate comparison tables and figures.

## Project structure

| File or folder | Purpose |
|---|---|
| `benchmark.ipynb` | Configure and run experiments |
| `run_benchmark.py` | Schedule experiments and manage recovery |
| `data_foundry/` | Prepare and inspect datasets |
| `models/` | Model wrappers and image-editing execution |
| `evaluators/` | Refusal detection and output grading |
| `metrics/` | Research metric definitions |
| `utils/` | Result handling, retries, tables, and plots |
| `tests/` | Offline correctness checks |

## Getting started

Use the dependency files in `environment/` to prepare your Python environment.

To run experiments:

1. Configure credentials for the API providers you intend to use.
2. Open `benchmark.ipynb`.
3. Set the model deployments and experiment settings.
4. Choose an existing run to resume, or prepare a new run.
5. Execute the notebook in order.

To inspect cases in a grid:

    inspect_cases(cases, columns=3, page_size=12, page=1)

## Saved results

Each run stores its dataset and experiment outputs. Results include generated
images, evaluation scores, and error information. Multi-step experiments also
save intermediate images and step records.

Rerunning an experiment skips completed evaluations and reuses generated images
when only grading needs to be retried.

Use a new run directory when changing the dataset, prompts, models, judge,
or evaluation settings.

## Current limitations

- Automated AI judge can be incorrect and require further validation.
- Refusal detection includes image-change heuristics, not only explicit refusals.
- Technical failures are excluded from scoring and reported separately.
