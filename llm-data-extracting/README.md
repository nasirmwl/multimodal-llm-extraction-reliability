# Code: *On the Reliability and Consistency of Multimodal LLMs for Structured Data Extraction*

This repository implements the **benchmark and analysis pipeline** for the paper draft in [`paper/draft.md`](paper/draft.md). Run `run_study.py` to regenerate all quantitative artifacts referenced there (`results/`).

## Paper scope (what this code measures)

We study **reliability and consistency** of structured extraction: not only average accuracy, but run-to-run variability, schema validity, line-item agreement, and robustness under **synthetic OCR perturbations**. Extractors see **OCR text and prompts only**; labels are used **only for scoring** (leakage-safe protocol, as in the paper).

**Multimodal caveat (Section 7 of the draft):** this workspace ships **OCR text + JSON labels**, not raw invoice images. The implemented task is therefore **OCR-to-JSON** reliability. Claims in the paper about full **image-conditioned** multimodal behavior require adding image inputs and a vision-capable extractor; the `Extractor` interface is the extension point.

## Task (Section 1 of the draft)

From OCR text, predict normalized JSON:

- **invoice:** `client_name`, `client_address`, `seller_name`, `seller_address`, `invoice_number`, `invoice_date`, `due_date`
- **items[]:** `description`, `quantity`, `total_price`

The **core score** excludes `due_date` when that field is non-informative in the labels (see draft).

## Dataset (Section 2 of the draft)

The paper draft assumes **1,414** labeled invoice rows (CSV batch). The `dataset/` directory is **gitignored**; obtain data separately and place CSVs under:

`dataset/batch_1/batch_1/`

Required columns: `File Name`, `Json Data`, `OCRed Text`.

## Experimental setup (Section 3 of the draft)

The code reproduces the draft’s design choices:

| Axis | Values in code |
|------|----------------|
| Prompt variants | `strict_json`, `few_shot_schema` |
| OCR stress variants | `clean`, `mild_perturb`, `heavy_perturb` (disable with `--no-stress-test`) |
| Repeated runs | `--runs` (default `5` matches draft) |
| Eval split | `--eval-fraction` (default `0.25`); `--seed` + `results/eval_split.json` for reproducibility |
| Metrics | Field exact/fuzzy, item F1, JSON validity, core score, run-level std, 95% CI, pairwise pseudo p-values |

## Models (Section 7 of the draft)

- **Offline:** `RegexBaselineExtractor`, `SimulatedLLMExtractor` — for pipeline validation; not sufficient alone for strong extraction claims in the paper.
- **Optional API:** `OpenAIGPT4oExtractor` when you pass `--use-production-gpt4o` and set `OPENAI_API_KEY` — use this (or another adapter you add) for **publishable** model numbers aligned with the paper’s multimodal / LLM angle, subject to the OCR-vs-image caveat above.

## Reproducibility (Section 8 of the draft)

Default run (matches draft repro block; offline extractors):

```bash
cd llm-data-extracting
python3 run_study.py --runs 5 --eval-fraction 0.25 --seed 42
```

Artifacts land under `results/` (gitignored). Key files cited in the draft:

- `benchmark_summary.md` / `benchmark_summary.json` — headline tables
- `stratified_performance.csv` — stratified analysis (Section 5)
- `significance_tests.csv` — pairwise comparisons
- `eval_split.json` — exact eval document IDs and seed

## Requirements

- **Python 3.9+** (stdlib for offline path).
- **OpenAI** (optional, for paper-quality LLM runs): `export OPENAI_API_KEY=...` then use `--use-production-gpt4o`. OCR text is sent to the API; use `--max-docs` for quota-safe pilots. See [`paper/draft.md`](paper/draft.md) limitations on scope.

## CLI (full reference)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | `dataset/batch_1/batch_1` | CSV directory. |
| `--output-root` | `results` | Output directory. |
| `--runs` | `5` | Repeated runs per configuration. |
| `--eval-fraction` | `0.25` | Eval fraction after shuffle. |
| `--seed` | `42` | Split + perturbation seed. |
| `--no-stress-test` | off | OCR variants: `clean` only. |
| `--use-production-gpt4o` | off | GPT-4o instead of local simulator for the LLM slot. |
| `--max-docs` | `0` | Cap eval docs (`0` = all). |
| `--fail-on-api-error` | off | Exit if API errors persist after retries. |

## Production implications (Section 6 of the draft)

The pipeline is built to support the paper’s recommendations: report **consistency** with means, enforce **no-leakage** evaluation, keep **JSON validators**, and use **stratified** monitoring (complexity + OCR stress in outputs).

## Project layout

```
llm-data-extracting/
  paper/draft.md            # Paper draft (sections align with this README)
  run_study.py
  src/reliability/
    schema.py data.py metrics.py models.py benchmark.py analysis.py
```
