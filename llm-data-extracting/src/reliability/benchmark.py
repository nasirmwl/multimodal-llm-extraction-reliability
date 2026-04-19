from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from .data import load_records, split_records
from .metrics import DocumentMetrics, evaluate_document, summarize_metrics
from .models import Extractor, OpenAIGPT4oExtractor, RegexBaselineExtractor, SimulatedLLMExtractor


PROMPT_VARIANTS = ("strict_json", "few_shot_schema")


def run_benchmark(
    dataset_root: Path,
    output_root: Path,
    runs: int = 5,
    eval_fraction: float = 0.25,
    seed: int = 42,
    run_stress_test: bool = True,
    use_production_gpt4o: bool = False,
    max_docs: int = 0,
    fail_on_api_error: bool = False,
) -> None:
    records = load_records(dataset_root)
    _, evaluate = split_records(records, eval_fraction=eval_fraction, seed=seed)
    if max_docs > 0:
        evaluate = evaluate[:max_docs]
    output_root.mkdir(parents=True, exist_ok=True)
    _write_eval_split(output_root, evaluate, seed=seed, eval_fraction=eval_fraction)

    llm_extractor: Extractor
    if use_production_gpt4o:
        llm_extractor = OpenAIGPT4oExtractor(fail_on_api_error=fail_on_api_error)
    else:
        llm_extractor = SimulatedLLMExtractor()
    extractors: list[Extractor] = [RegexBaselineExtractor(), llm_extractor]
    ocr_variants = ("clean", "mild_perturb", "heavy_perturb") if run_stress_test else ("clean",)
    summary_rows: list[dict] = []
    all_scores: dict[tuple[str, str, str], list[float]] = {}
    detail_path = output_root / "benchmark_details.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "prompt_variant",
                "ocr_variant",
                "run_id",
                "file_name",
                "exact_field_accuracy",
                "fuzzy_field_accuracy",
                "item_f1",
                "json_valid",
                "core_score",
            ],
        )
        writer.writeheader()

        for extractor in extractors:
            for prompt_variant in PROMPT_VARIANTS:
                for ocr_variant in ocr_variants:
                    run_metrics: list[list[DocumentMetrics]] = []
                    for run_id in range(runs):
                        per_doc: list[DocumentMetrics] = []
                        for record in evaluate:
                            perturbed_ocr = _apply_ocr_variant(
                                record.ocr_text,
                                ocr_variant=ocr_variant,
                                seed=seed + run_id + abs(hash(record.file_name)) % 10_000,
                            )
                            pred = extractor.extract(
                                ocr_text=perturbed_ocr,
                                prompt_variant=prompt_variant,
                                run_seed=run_id + seed,
                            )
                            doc_metrics = evaluate_document(pred, record.ground_truth)
                            per_doc.append(doc_metrics)
                            writer.writerow(
                                {
                                    "model": extractor.name,
                                    "prompt_variant": prompt_variant,
                                    "ocr_variant": ocr_variant,
                                    "run_id": run_id,
                                    "file_name": record.file_name,
                                    **asdict(doc_metrics),
                                }
                            )
                        run_metrics.append(per_doc)
                    aggregate = summarize_metrics(run_metrics)
                    summary_row = {
                        "model": extractor.name,
                        "prompt_variant": prompt_variant,
                        "ocr_variant": ocr_variant,
                        **aggregate,
                    }
                    run_level_core_scores = [mean([d.core_score for d in run]) for run in run_metrics]
                    summary_row.update(
                        _confidence_interval_95(run_level_core_scores)
                    )
                    summary_rows.append(summary_row)
                    all_scores[(extractor.name, prompt_variant, ocr_variant)] = run_level_core_scores

    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    _write_high_level_markdown(summary_rows, output_root / "benchmark_summary.md")
    _write_significance(all_scores, output_root / "significance_tests.csv")


def _write_high_level_markdown(rows: Iterable[dict], out_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda x: x.get("core_score_mean", 0), reverse=True)
    lines = [
        "# Benchmark Summary",
        "",
        "| Model | Prompt | OCR Variant | Core Score (mean±std) | 95% CI | Exact Fields | Item F1 | JSON Valid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted_rows:
        lines.append(
            "| {model} | {prompt_variant} | {ocr_variant} | {core_score_mean:.4f} ± {core_score_std:.4f} | "
            "[{core_score_ci95_low:.4f}, {core_score_ci95_high:.4f}] | "
            "{exact_field_accuracy_mean:.4f} | {item_f1_mean:.4f} | {json_valid_rate_mean:.4f} |".format(
                **row
            )
        )
    if sorted_rows:
        top = sorted_rows[0]
        lines.extend(
            [
                "",
                "Top configuration:",
                f"- `{top['model']}` + `{top['prompt_variant']}`",
                f"- mean core score `{top['core_score_mean']:.4f}` across repeated runs",
                f"- run-to-run std `{top['core_score_std']:.4f}`",
            ]
        )
        lines.extend(
            [
                "",
                "Average across all configurations:",
                f"- core score mean `{mean([r['core_score_mean'] for r in sorted_rows]):.4f}`",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_eval_split(
    output_root: Path, evaluate: list, seed: int, eval_fraction: float
) -> None:
    payload = {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "eval_files": [r.file_name for r in evaluate],
    }
    (output_root / "eval_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_ocr_variant(text: str, ocr_variant: str, seed: int) -> str:
    if ocr_variant == "clean":
        return text
    rng = random.Random(seed)
    chars = list(text)
    if ocr_variant == "mild_perturb":
        drop_prob = 0.01
        swap_prob = 0.01
    else:
        drop_prob = 0.03
        swap_prob = 0.03
    out: list[str] = []
    for ch in chars:
        if rng.random() < drop_prob:
            continue
        if ch.isdigit() and rng.random() < swap_prob:
            out.append(str(rng.randint(0, 9)))
            continue
        if ch.isalpha() and rng.random() < swap_prob:
            out.append(ch.upper() if ch.islower() else ch.lower())
            continue
        out.append(ch)
    return "".join(out)


def _confidence_interval_95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"core_score_ci95_low": 0.0, "core_score_ci95_high": 0.0}
    m = mean(values)
    if len(values) == 1:
        return {"core_score_ci95_low": m, "core_score_ci95_high": m}
    # normal approximation on run-level means
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    std = variance**0.5
    margin = 1.96 * std / (len(values) ** 0.5)
    return {"core_score_ci95_low": m - margin, "core_score_ci95_high": m + margin}


def _write_significance(
    all_scores: dict[tuple[str, str, str], list[float]], out_path: Path
) -> None:
    keys = sorted(all_scores.keys())
    rows = []
    for idx, key_a in enumerate(keys):
        for key_b in keys[idx + 1 :]:
            a = all_scores[key_a]
            b = all_scores[key_b]
            if len(a) != len(b):
                continue
            diffs = [x - y for x, y in zip(a, b)]
            wins = sum(1 for d in diffs if d > 0)
            losses = sum(1 for d in diffs if d < 0)
            non_ties = wins + losses
            if non_ties == 0:
                p_value = 1.0
            else:
                win_rate = wins / non_ties
                p_value = 2 * min(win_rate, 1 - win_rate)
            rows.append(
                {
                    "a_model": key_a[0],
                    "a_prompt": key_a[1],
                    "a_ocr_variant": key_a[2],
                    "b_model": key_b[0],
                    "b_prompt": key_b[1],
                    "b_ocr_variant": key_b[2],
                    "mean_diff_core_score": round(mean(diffs), 6) if diffs else 0.0,
                    "pseudo_p_value": round(p_value, 6),
                }
            )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("a_model,a_prompt,a_ocr_variant,b_model,b_prompt,b_ocr_variant,mean_diff_core_score,pseudo_p_value\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

