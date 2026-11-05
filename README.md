# BLINDSPOT

An image editing benchmark for refusal, edit fidelity, and realism.

## Setup

Use Python 3.11+ and run commands from the project root. Extract the dataset to `data/benchmark_cases`.

```bash
python -m pip install -r environment/requirements.txt
```

Set credentials for the providers you use.

| Provider | Credentials |
| --- | --- |
| Azure | `AZURE_OPENAI_BASE_URL`, `Azure_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` and Google application default credentials |

For local generation, use Linux with a BF16-capable NVIDIA GPU and CUDA-compatible PyTorch. Install local dependencies and accept any required model access terms.

```bash
python -m pip install -r environment/local_requirements.txt
hf auth login
```

## Generate

Edit [configs/benchmark.yaml](configs/benchmark.yaml) to set models, judge, and levels. Azure model names are deployment names. Command line options override the config.

```bash
python run.py --config configs/benchmark.yaml --models sunburst --output runs/sunburst
python run.py --config configs/benchmark.yaml --models qwen-image-2.1 flux-3-klein flux-2-dev --output runs/local
```

Add `--check` to validate without API calls, or `--smoke-test --levels L1` to try one case in a fresh output folder. API L3 generation uses the configured judge to decide whether to continue each chain.

Repeat the generation command with the same output folder and settings to resume. Completed cases are skipped and failed cases are retried. For a 10% pilot, add `--run-percent 10`, then remove it to continue the full run in the same folder.

## Judge

Judge saved outputs after generation finishes. The config selects Sol for fidelity and realism. Replace `HAIKU_DEPLOYMENT` with your Azure deployment name for refusal judging.

```bash
python run.py --config configs/benchmark.yaml --mode judge --models sunburst --output runs/sunburst --refusal-judge-model HAIKU_DEPLOYMENT --max-workers 4
python run.py --config configs/benchmark.yaml --mode judge --output runs/local --skip-refusal --max-workers 4
```

Use `--skip-refusal` for local outputs. API L3 keeps the refusal decisions recorded during generation. Realism is scored from 1 to 5 when fidelity passes. The configured GPT-6 Sol deployment uses Flex.

Judging saves a separate `OUTPUT-scored` folder. Repeat the judging command with `--resume` to retry failed or unfinished judgments with the same settings. Use `--mode all` to generate and judge in one command.

Open `benchmark.ipynb`, set `run_dir` to the scored folder, and run the review section. See `python run.py --help` for options or `python judge_results.py --help` for calibration against human labels.
