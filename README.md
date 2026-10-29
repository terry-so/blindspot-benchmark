# BLINDSPOT

An image-editing benchmark measuring refusal, edit fidelity, and realism.
Results are preliminary.

## Setup

Use Python 3.11+. Run commands from the project folder. Make sure
`data/benchmark_cases` is present; it is not included in a Git clone.

For API models and notebooks:

```bash
pip install -r environment/requirements.txt
```

For local GPU models, follow [LOCAL_MODELS.md](LOCAL_MODELS.md).

## Configure

Edit `configs/benchmark.yaml`:

- Set the model IDs and judge deployment. Azure uses deployment names.
- Choose the prompt levels, repeats, and output folder.
- Set `cases_per_subcategory: 2` for 60 images, or omit it to use all cases.

Set credentials in your terminal environment for the selected providers and judge:

- Azure: `AZURE_OPENAI_BASE_URL` and `Azure_API_KEY`.
- OpenAI: `OPENAI_API_KEY`.
- Gemini: `GEMINI_API_KEY`.
- Vertex AI: `GOOGLE_CLOUD_PROJECT` and Google application-default credentials.

Local models currently use an Azure judge.

## Run

Check one model's setup:

```bash
python run.py --config configs/benchmark.yaml --models flare --check
```

For API models, this checks configuration and files, not API access.

Test one image with L1:

```bash
python run.py --config configs/benchmark.yaml --models flare --levels L1 --smoke-test --output runs/test-flare
```

This makes real generation and judge calls.

Run selected models:

```bash
python run.py --config configs/benchmark.yaml --models flare sunburst
```

Omit `--models` to run all models listed in the YAML. API models share
`max_workers`; local models run one at a time. CLI arguments override shared
YAML settings. Use `python run.py --help` for options.

## Resume

Rerun the same command with the same output folder. Completed cases are skipped;
saved images are reused when only judging failed.

Use a new output folder when changing settings or case selection. Existing runs
reuse their saved dataset, so edits to the original dataset do not affect them.

## Results

For the multi-model config:

- Original images and case metadata: `OUTPUT/data/`.
- API outputs and scores: `OUTPUT/results/MODEL_NAME/`.
- Local outputs and scores: `OUTPUT/local/MODEL_NAME/results/LOCAL_MODEL/`.

Open `benchmark.ipynb` to preview cases, run the same pipeline, compare before/after
images, and create tables and plots. To review an existing run, set `run_dir` to
its output folder and run the review section only.

Scores are automated judge estimates. Realism is scored only when fidelity
succeeds; technical failures are reported separately. LPIPS is not used for refusal.

