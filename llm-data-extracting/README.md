# Invoice Extraction Reliability Benchmark

This project evaluates reliability and consistency for OCR-to-JSON invoice extraction.

## Dataset

The `dataset/` directory is gitignored. Place CSVs under `dataset/batch_1/batch_1/` (same layout as before) or pass `--dataset-root` to `run_study.py`.

- Input: `OCRed Text`
- Target: `Json Data` canonicalized to:
  - `invoice`: `client_name`, `client_address`, `seller_name`, `seller_address`, `invoice_number`, `invoice_date`, `due_date`
  - `items[]`: `description`, `quantity`, `total_price`

## What This Pipeline Produces

- `results/benchmark_details.csv`: per-document, per-run metrics.
- `results/benchmark_summary.json`: aggregate metrics by model/prompt.
- `results/benchmark_summary.md`: leaderboard and consistency stats.
- `results/eval_split.json`: reproducible evaluation set file IDs and seed.
- `results/document_profiles.csv`: OCR-noise and complexity buckets.
- `results/analysis_summary.json`: bucket distributions and dataset profile.
- `results/stratified_performance.csv`: stratified performance by model/prompt/OCR variant/complexity.
- `results/significance_tests.csv`: pairwise sign-test-style comparisons.

## Metrics

- Field exact accuracy
- Field fuzzy accuracy (string similarity)
- Item-level F1 (exact item tuple match)
- JSON schema validity rate
- Core score = average of:
  - invoice exact accuracy excluding `due_date`
  - item F1
  - JSON validity
- Consistency via run-to-run standard deviation for each metric
- 95% confidence interval for core score (run-level mean CI)

## Run

```bash
python3 run_study.py --runs 5 --eval-fraction 0.25 --seed 42
```

Optional:

```bash
python3 run_study.py --runs 5 --eval-fraction 0.25 --seed 42 --no-stress-test
```

## Hardening Notes

- No label leakage: extractors only receive OCR text, prompt variant, and run seed.
- Reproducible split: deterministic shuffled split controlled by `--seed`.
- OCR robustness stress: `clean`, `mild_perturb`, and `heavy_perturb` OCR variants.
- Current included extractors are offline baselines/simulators. Add real API-backed LLM extractors for publishable model comparisons.

