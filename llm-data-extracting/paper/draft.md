# On the Reliability and Consistency of Multimodal LLMs for Structured Data Extraction

## Abstract

We study reliability and consistency for structured invoice extraction from OCR text. The benchmark uses 1,414 labeled documents and evaluates field-level correctness, line-item matching, schema validity, run-to-run variability, and OCR perturbation robustness under repeated inference. We enforce a leakage-safe protocol where extractors only access OCR text and prompts. Current offline baselines underperform, highlighting the need for API-backed multimodal LLM evaluation before drawing model-level claims.

## 1. Task Definition

Given OCR text for an invoice, predict normalized JSON with:

- `invoice`: `client_name`, `client_address`, `seller_name`, `seller_address`, `invoice_number`, `invoice_date`, `due_date`
- `items[]`: `description`, `quantity`, `total_price`

The benchmark excludes `due_date` from the core score because this field is non-informative in the current labels (empty across all examples).

## 2. Dataset

- Total records: 1,414
- Evaluation split: 25% (353 documents)
- Average line-item count in eval: 4.01
- OCR noise profile: mostly low-noise OCR in this batch
- Complexity mix (eval): 99 simple, 152 moderate, 102 complex documents

## 3. Experimental Setup

- Prompt variants:
  - `strict_json`
  - `few_shot_schema`
- OCR variants:
  - `clean`
  - `mild_perturb`
  - `heavy_perturb`
- Repeated runs: 5 per model/prompt setup
- Metrics:
  - field exact accuracy
  - field fuzzy accuracy
  - item-level F1
  - JSON validity rate
  - core score = mean(non-`due_date` field exact, item F1, JSON validity)
- Reliability is measured by run-to-run standard deviation (`std`) of each metric.
- Core score confidence intervals are reported using run-level 95% CI.
- Pairwise comparisons are reported with sign-test-style pseudo p-values.

## 4. Main Results

From `results/benchmark_summary.md`:

- `regex_baseline` variants: core score `0.3333 ± 0.0000`
- `simulated_llm` variants: core score `0.3333 ± 0.0000`
- All pairwise differences are effectively zero in current offline setup.

Key observation: after leakage removal, current offline extractors do not recover invoice structure from OCR text robustly; benchmark hardening is successful, but model quality is insufficient for meaningful comparative claims.

## 5. Stratified Failure Analysis

From `results/stratified_performance.csv`:

- Scores remain flat and low across complexity buckets and OCR variants for all included extractors.
- This indicates failure modes are fundamental extraction limitations, not just stress-condition brittleness.

## 6. Reliability Implications for Production

- Report consistency (`std`) alongside mean accuracy; average-only reporting can mask instability.
- Enforce leakage-safe protocols (no label access inside extractors) before any model reporting.
- Keep JSON validators in the serving loop even when validity rates are high.
- Use stratified monitoring by complexity and OCR quality to catch silent regressions.

## 7. Limitations

- This workspace currently includes OCR text + labels but not raw invoice images, so this benchmark is OCR-to-JSON rather than full multimodal image-to-JSON.
- Current model comparison includes one deterministic regex baseline and one offline stochastic simulator; both are insufficient for strong extraction. Live API-backed multimodal LLMs should be plugged into the existing extractor interface for final claims.

## 8. Reproducibility

Run:

```bash
python3 run_study.py --runs 5 --eval-fraction 0.25 --seed 42
```

Outputs are written under `results/`.

