# BLINDSPOT

An image-editing benchmark measuring refusal, edit fidelity, and realism.
Results are preliminary.

## Setup

Use Python 3.11+ and run commands from the project root. Extract the dataset
so `data/benchmark_cases` exists; it is not included in a Git clone.

For API models:

```bash
python -m pip install -r environment/requirements.txt
```

For local models, use Linux with an NVIDIA GPU supporting BF16 and install
CUDA-compatible PyTorch first. Then install dependencies and log in to Hugging Face:

```bash
python -m pip install -r environment/local_requirements.txt
hf auth login
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
```

Both Python checks must print `True`. Accept any required model access terms
on Hugging Face. Local memory requirements depend on the model and CPU offload.

Set credentials for the API providers you use:

- Azure: `AZURE_OPENAI_BASE_URL` and `Azure_API_KEY`.
- OpenAI: `OPENAI_API_KEY`.
- Gemini: `GEMINI_API_KEY`.
- Vertex AI: `GOOGLE_CLOUD_PROJECT` and Google application-default credentials.

## Configuration

Edit `configs/benchmark.yaml`: choose models, judge, levels, repeats, output,
and `max_workers`. Azure model IDs are deployment names. Local models have
`cpu_offload: true` configured. CLI options override YAML settings.

Use `--models` to select names from the YAML; omit it to run every listed model.
API generation shares `max_workers`; local models run sequentially.

## Run

Generate edits (`generate` is the default):

```bash
python run.py --config configs/benchmark.yaml --models sunburst --mode generate --output runs/sunburst
```

Judge saved edits:

```bash
python run.py --config configs/benchmark.yaml --models sunburst --mode judge --output runs/sunburst
```

Generate and then judge:

```bash
python run.py --config configs/benchmark.yaml --models sunburst --mode all --output runs/sunburst
```

Generate all local models without external judging:

```bash
python run.py --config configs/benchmark.yaml --models qwen-image-2.1 flux-3-klein flux-2-dev --mode generate --output runs/local-generation
```

Add `--check` to validate without API calls. Add `--levels L1 --smoke-test`
to test one case per selected model in a separate output folder.

API L3 generation needs the judge for refusal checks, which control whether
editing continues. Local image-only models use response/error checks instead.
Fidelity and realism are evaluated afterward. `all` stops before judging if
generation reports errors. See `python run.py --help` for remaining options.

## Pilot and resume

Run 10% of cases, balanced across subcategories:

```bash
python run.py --config configs/benchmark.yaml --models sunburst --mode generate --run-percent 10 --output runs/sunburst-full
```

Keep `cases_per_subcategory` unset. With 600 cases, 20 per subcategory, this
runs 60 cases at all configured levels and saves the full dataset for continuation.

After reviewing the cost, continue in the same folder:

```bash
python run.py --config configs/benchmark.yaml --models sunburst --mode generate --output runs/sunburst-full
```

Completed edits are skipped. Repeat the same generation command to resume;
keep model, judge, levels, repeats, and worker settings unchanged. Use a new
folder when changing settings or dataset selection.

## Results

`--output` identifies the generation run. Judging copies it to `OUTPUT-scored`;
use `--judged-output` to choose another new folder. L3 refusal decisions are
preserved. Judge only after generation has stopped; no GPU is needed for judging.

For runs using the YAML model list:

- `OUTPUT/data/`: original images and case metadata.
- `OUTPUT/results/MODEL_NAME/`: API edited images and results.
- `OUTPUT/local/MODEL_NAME/results/LOCAL_MODEL/`: local edited images and results.

Return the complete output folder when sharing results. Open `benchmark.ipynb`,
set `run_dir`, and run its review section to inspect images and scores.
Realism is scored only when fidelity succeeds; technical failures are separate.

For calibration against human labels, regrade metrics independently:

```bash
python judge_results.py --run runs/sunburst-scored --output runs/calibration --metrics refusal fidelity realism --labels labels.csv --judge-model YOUR_DEPLOYMENT --max-workers 4
```

Choose one or more metrics. The CSV needs `output_file`, `case_id`, `model`,
`level`, and the corresponding columns: `human_refused` (0/1),
`human_fidelity` (0/1), and `human_realism` (1–5). Each metric uses cases with
both a human and an existing AI score; missing-label counts are reported.
Without `--labels`, all eligible existing scores are regraded without comparison.

Each metric gets its own folder containing `scores.csv`, `agreement.json`,
checkpoints, and the prompts used. Source results stay unchanged. Reports
include agreement, kappa, directional errors, and model/level/subcategory
breakdowns; realism also includes within-one agreement and quadratic weighted kappa.

Add `--disagreements-only` for diagnosis, `--check` to validate without calls,
or `--resume` to continue with the same settings and prompts. New scores do
not gate other metrics. L3 refusal calibration covers recorded steps only;
it cannot reconstruct edits skipped after an earlier refusal. Validate the
final judge on a separate held-out sample. See `python judge_results.py --help`.

Generation results include `estimated_cost_usd` for each edit and L3 step.
To enable estimates for Azure/OpenAI image models, add `pricing` under the
model entry in the YAML: `{text_input: 5, image_input: 8, output: 30}`.
Flare/Sunburst use these earlier pilot assumptions in the YAML. Rates are USD
per million tokens; confirm your deployment prices.
`cost_summary.json` beside each `results/` directory totals recorded generation
cost per model when generation finishes. Repeating a run recalculates totals
without double-counting completed edits. Changing pricing recalculates estimates
for selected records visited by that invocation. The main run summary also
includes local models, whose GPU costs remain unknown.

Missing/unsupported usage, cached-token usage, and local GPU costs remain `null`.
Totals include priced calls only; `unpriced_calls` shows missing estimates.
Judge costs, unlogged retries, and overwritten attempts are excluded.
