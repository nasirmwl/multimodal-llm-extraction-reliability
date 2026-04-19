from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from .data import load_records, split_records


def ocr_noise_score(text: str) -> float:
    if not text:
        return 1.0
    printable = sum(ch.isprintable() for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    ratio_printable = printable / len(text)
    ratio_alnum = (alpha + digits) / len(text)
    # Lower value = noisier; convert to "noise severity" (higher is noisier).
    return 1.0 - (0.6 * ratio_printable + 0.4 * ratio_alnum)


def _bucket_noise(score: float) -> str:
    if score < 0.15:
        return "low_noise"
    if score < 0.30:
        return "medium_noise"
    return "high_noise"


def _bucket_complexity(item_count: int) -> str:
    if item_count <= 2:
        return "simple"
    if item_count <= 5:
        return "moderate"
    return "complex"


def run_failure_analysis(dataset_root: Path, output_root: Path) -> None:
    records = load_records(dataset_root)
    eval_fraction = _read_eval_fraction(output_root)
    seed = _read_seed(output_root)
    _, evaluate = split_records(records, eval_fraction=eval_fraction, seed=seed)
    output_root.mkdir(parents=True, exist_ok=True)

    profile_rows = []
    for record in evaluate:
        item_count = len(record.ground_truth.get("items", []))
        noise = ocr_noise_score(record.ocr_text)
        profile_rows.append(
            {
                "file_name": record.file_name,
                "item_count": item_count,
                "ocr_noise_score": round(noise, 4),
                "noise_bucket": _bucket_noise(noise),
                "complexity_bucket": _bucket_complexity(item_count),
            }
        )

    profile_path = output_root / "document_profiles.csv"
    with profile_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    summary = {
        "num_eval_docs": len(profile_rows),
        "avg_item_count": mean([r["item_count"] for r in profile_rows]),
        "avg_ocr_noise_score": mean([r["ocr_noise_score"] for r in profile_rows]),
        "noise_bucket_distribution": _distribution(profile_rows, "noise_bucket"),
        "complexity_bucket_distribution": _distribution(profile_rows, "complexity_bucket"),
    }
    (output_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_stratified_performance(output_root, profile_rows)


def _distribution(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return out


def _write_stratified_performance(output_root: Path, profile_rows: list[dict]) -> None:
    details_path = output_root / "benchmark_details.csv"
    if not details_path.exists():
        return

    profile_by_file = {row["file_name"]: row for row in profile_rows}
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row["file_name"]
            profile = profile_by_file.get(file_name)
            if profile is None:
                continue
            key = (
                row["model"],
                row["prompt_variant"],
                row.get("ocr_variant", "clean"),
                profile["complexity_bucket"],
            )
            grouped.setdefault(key, []).append(float(row["core_score"]))

    out_rows = []
    for (model, prompt_variant, ocr_variant, complexity_bucket), scores in grouped.items():
        out_rows.append(
            {
                "model": model,
                "prompt_variant": prompt_variant,
                "ocr_variant": ocr_variant,
                "complexity_bucket": complexity_bucket,
                "count": len(scores),
                "core_score_mean": round(mean(scores), 4),
            }
        )
    out_rows.sort(
        key=lambda r: (
            r["model"],
            r["prompt_variant"],
            r["ocr_variant"],
            r["complexity_bucket"],
        )
    )
    if not out_rows:
        return

    out_path = output_root / "stratified_performance.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def _read_eval_fraction(output_root: Path) -> float:
    split_path = output_root / "eval_split.json"
    if not split_path.exists():
        return 0.25
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return float(payload.get("eval_fraction", 0.25))


def _read_seed(output_root: Path) -> int:
    split_path = output_root / "eval_split.json"
    if not split_path.exists():
        return 42
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return int(payload.get("seed", 42))

