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

## Measure API generation cost

New Flare/Sunburst results include token usage when the API returns it. Run a small pilot:

```bash
python run.py --config configs/benchmark.yaml --models flare sunburst --levels L1 --smoke-test --output runs/cost-pilot-flare-sunburst
python estimate_image_cost.py runs/cost-pilot-flare-sunburst --planned-edits 4200
```

The report defaults to USD per million tokens: text input 5, image input 8,
image output 30. Override with `--text-rate`, `--image-rate`, and `--output-rate`
to match your Azure deployment. Judge costs and unlogged retries are excluded;
missing usage is unknown cost. Check Azure billing for actual charges.
The 4,200-call projection assumes 600 cases, L0/L1/L2a/L2b, and three L3 steps
per case, with one repeat and no early stopping. A one-case L1 pilot is only an
initial measurement; sample all levels and subcategories before fixing a budget.
