# Invoice OCR extraction reliability benchmark

Benchmark for **structured invoice extraction** from OCR text: field accuracy, line-item F1, JSON validity, **run-to-run consistency**, OCR stress variants, and simple pairwise comparisons. No labels are passed into extractors.

## Requirements

- **Python 3.9+** (stdlib only for the default offline path).
- **OpenAI** (optional): set `OPENAI_API_KEY` when using `--use-production-gpt4o`.

## Dataset

The `dataset/` directory is **gitignored**. Obtain CSVs separately and place them under:

`dataset/batch_1/batch_1/`

Each CSV must have columns: `File Name`, `Json Data` (ground-truth JSON string), `OCRed Text`.

Canonical schema after normalization:

- **invoice:** `client_name`, `client_address`, `seller_name`, `seller_address`, `invoice_number`, `invoice_date`, `due_date`
- **items[]:** `description`, `quantity`, `total_price`

`due_date` is excluded from the **core score** when labels are uniformly empty (non-informative).

## Quick start (offline, no API)

```bash
cd llm-data-extracting
python3 run_study.py --runs 5 --eval-fraction 0.25 --seed 42
```

Uses `RegexBaselineExtractor` + `SimulatedLLMExtractor` only. OCR goes to the model path; ground truth is used **only** for scoring.

## OpenAI GPT-4o (production extractor)

1. Set the key in your shell (do not commit keys):

   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

2. Start with a small run to avoid rate limits (`429`):

   ```bash
   python3 run_study.py --use-production-gpt4o --runs 1 --eval-fraction 0.05 --max-docs 20 --seed 42
   ```

3. Scale up when stable:

   ```bash
   python3 run_study.py --use-production-gpt4o --runs 5 --eval-fraction 0.25 --seed 42
   ```

- **`--max-docs N`:** cap eval documents (`0` = all). Helps with quotas and 429s.
- **`--fail-on-api-error`:** exit if the API keeps failing after retries (default is to score empty predictions and continue).
- Enabling GPT mode **sends OCR text to OpenAI**; treat that as external data processing for privacy/compliance.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | `dataset/batch_1/batch_1` | Directory containing `*.csv` files. |
| `--output-root` | `results` | Where artifacts are written. |
| `--runs` | `5` | Repeated runs per model × prompt × OCR variant. |
| `--eval-fraction` | `0.25` | Fraction of records used for evaluation (after shuffle). |
| `--seed` | `42` | Split + perturbation RNG seed. |
| `--no-stress-test` | off | If set, only `clean` OCR (no `mild_perturb` / `heavy_perturb`). |
| `--use-production-gpt4o` | off | Use OpenAI GPT-4o instead of the local simulator for the “LLM” slot. |
| `--max-docs` | `0` | Max eval documents (`0` = no cap). |
| `--fail-on-api-error` | off | Fail hard on persistent API errors. |

## Outputs (`results/`)

| File | Purpose |
|------|---------|
| `benchmark_details.csv` | Per document, run, model, prompt, OCR variant: metrics. |
| `benchmark_summary.json` | Aggregated means/std and 95% CI on core score. |
| `benchmark_summary.md` | Human-readable table. |
| `eval_split.json` | `seed`, `eval_fraction`, list of eval `file_name`s (reproducibility). |
| `document_profiles.csv` | Per-doc complexity and heuristic OCR-noise score. |
| `analysis_summary.json` | Counts and bucket distributions. |
| `stratified_performance.csv` | Mean core score by model / prompt / OCR variant / complexity. |
| `significance_tests.csv` | Pairwise comparisons of run-level core scores (sign-test-style pseudo p-values). |

`results/` is gitignored; regenerate with `run_study.py`.

## Metrics

- **Field exact / fuzzy** accuracy over canonical invoice fields.
- **Item F1** multiset match on `(description, quantity, total_price)` tuples.
- **JSON validity** strict schema check.
- **Core score:** mean of (invoice exact excluding `due_date`, item F1, JSON validity).
- **Consistency:** std across runs; **95% CI** on run-level mean core score.

## Project layout

```
llm-data-extracting/
  run_study.py              # CLI entrypoint
  src/reliability/
    schema.py               # Normalization + record type
    data.py                 # CSV load + seeded split
    metrics.py              # Scoring + aggregation
    models.py               # Regex baseline, simulator, optional GPT-4o
    benchmark.py            # Main loop + OCR stress + summaries
    analysis.py             # Profiles + stratified tables
  paper/draft.md            # Draft write-up (optional)
```

## Design notes

- **No leakage:** `extract(ocr_text, prompt_variant, run_seed)` only; labels used in `evaluate_document` only.
- **Reproducibility:** same `--seed` and `--eval-fraction` + saved `eval_split.json` align benchmark and analysis.
- **Stress test:** synthetic OCR drops/swap noise on `mild_perturb` and `heavy_perturb` (unless `--no-stress-test`).
