from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean, pstdev
from typing import Any

from .schema import CANONICAL_FIELDS, EXCLUDED_FIELDS_FROM_CORE, ITEM_FIELDS


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _safe_items(doc: dict[str, Any]) -> list[dict[str, str]]:
    items = doc.get("items", [])
    return items if isinstance(items, list) else []


def json_schema_valid(pred: dict[str, Any]) -> bool:
    if not isinstance(pred, dict):
        return False
    inv = pred.get("invoice")
    items = pred.get("items")
    if not isinstance(inv, dict) or not isinstance(items, list):
        return False
    return all(field in inv for field in CANONICAL_FIELDS) and all(
        isinstance(item, dict) and all(k in item for k in ITEM_FIELDS) for item in items
    )


@dataclass(frozen=True)
class DocumentMetrics:
    exact_field_accuracy: float
    fuzzy_field_accuracy: float
    item_f1: float
    json_valid: float
    core_score: float


def _items_to_multiset(items: list[dict[str, str]]) -> Counter:
    keys = []
    for item in items:
        keys.append(tuple(item.get(field, "") for field in ITEM_FIELDS))
    return Counter(keys)


def item_f1_score(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    pred_set = _items_to_multiset(_safe_items(pred))
    gt_set = _items_to_multiset(_safe_items(gt))
    matched = sum((pred_set & gt_set).values())
    pred_total = sum(pred_set.values())
    gt_total = sum(gt_set.values())
    if pred_total == 0 and gt_total == 0:
        return 1.0
    if pred_total == 0 or gt_total == 0:
        return 0.0
    precision = matched / pred_total
    recall = matched / gt_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_document(pred: dict[str, Any], gt: dict[str, Any]) -> DocumentMetrics:
    pred_inv = pred.get("invoice", {}) if isinstance(pred, dict) else {}
    gt_inv = gt.get("invoice", {})
    exact_scores: list[float] = []
    fuzzy_scores: list[float] = []
    for field in CANONICAL_FIELDS:
        pred_val = str(pred_inv.get(field, "")).strip()
        gt_val = str(gt_inv.get(field, "")).strip()
        exact_scores.append(1.0 if pred_val == gt_val else 0.0)
        fuzzy_scores.append(text_similarity(pred_val, gt_val))
    exact_field_accuracy = mean(exact_scores) if exact_scores else 0.0
    fuzzy_field_accuracy = mean(fuzzy_scores) if fuzzy_scores else 0.0
    item_f1 = item_f1_score(pred, gt)
    json_valid = 1.0 if json_schema_valid(pred) else 0.0

    core_fields = [f for f in CANONICAL_FIELDS if f not in EXCLUDED_FIELDS_FROM_CORE]
    core_exact = []
    for field in core_fields:
        pred_val = str(pred_inv.get(field, "")).strip()
        gt_val = str(gt_inv.get(field, "")).strip()
        core_exact.append(1.0 if pred_val == gt_val else 0.0)
    core_field_accuracy = mean(core_exact) if core_exact else 0.0
    core_score = mean([core_field_accuracy, item_f1, json_valid])
    return DocumentMetrics(
        exact_field_accuracy=exact_field_accuracy,
        fuzzy_field_accuracy=fuzzy_field_accuracy,
        item_f1=item_f1,
        json_valid=json_valid,
        core_score=core_score,
    )


def summarize_metrics(
    all_runs: list[list[DocumentMetrics]],
) -> dict[str, float]:
    if not all_runs:
        return {}
    per_run_means: dict[str, list[float]] = defaultdict(list)
    for run in all_runs:
        if not run:
            continue
        per_run_means["exact_field_accuracy"].append(
            mean([m.exact_field_accuracy for m in run])
        )
        per_run_means["fuzzy_field_accuracy"].append(
            mean([m.fuzzy_field_accuracy for m in run])
        )
        per_run_means["item_f1"].append(mean([m.item_f1 for m in run]))
        per_run_means["json_valid_rate"].append(mean([m.json_valid for m in run]))
        per_run_means["core_score"].append(mean([m.core_score for m in run]))

    out: dict[str, float] = {}
    for key, values in per_run_means.items():
        out[f"{key}_mean"] = mean(values)
        out[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return out

