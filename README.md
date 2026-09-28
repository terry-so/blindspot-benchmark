# BLINDSPOT Image Editing Benchmark

**Status: Work in progress.** Currently generates image-editing cases and evaluates model responses. Results are experimental.

## How to Run

1. Install the project dependencies and set `GEMINI_API_KEY`.
2. Open `benchmark.ipynb` from the repository root.
3. Set `cases_per_subcategory` and `use_reviewed`, then run the four steps: generate, inspect, benchmark, create tables.

Each run saves its cases, results, and CSVs under `runs/`.

To add a reviewed case, copy its folder (including `case.json` and `input.png`) into `data/reviewed/`, then run:

```python
from data_foundry.generate_seed import update_reviewed_dataset
update_reviewed_dataset()
```
L3 chain evaluation is not implemented yet.
