from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .schema import InvoiceRecord, canonicalize_invoice


def load_records(dataset_root: Path) -> list[InvoiceRecord]:
    records: list[InvoiceRecord] = []
    for csv_path in sorted(dataset_root.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                payload = json.loads(row["Json Data"])
                records.append(
                    InvoiceRecord(
                        file_name=row["File Name"],
                        ocr_text=row["OCRed Text"],
                        ground_truth=canonicalize_invoice(payload),
                    )
                )
    return records


def split_records(
    records: list[InvoiceRecord], eval_fraction: float = 0.25, seed: int = 42
) -> tuple[list[InvoiceRecord], list[InvoiceRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(records)
    eval_count = max(1, int(total * eval_fraction))
    train = shuffled[:-eval_count]
    evaluate = shuffled[-eval_count:]
    return train, evaluate

