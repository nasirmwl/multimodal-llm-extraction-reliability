from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


CANONICAL_FIELDS = (
    "client_name",
    "client_address",
    "seller_name",
    "seller_address",
    "invoice_number",
    "invoice_date",
    "due_date",
)

ITEM_FIELDS = ("description", "quantity", "total_price")
EXCLUDED_FIELDS_FROM_CORE = {"due_date"}


@dataclass(frozen=True)
class InvoiceRecord:
    file_name: str
    ocr_text: str
    ground_truth: dict[str, Any]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_number(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = text.replace(",", "")
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def _normalize_date(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    # Keep light normalization to avoid overfitting assumptions.
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", text)
    if match:
        mm, dd, yyyy = match.groups()
        return f"{int(mm):02d}/{int(dd):02d}/{yyyy}"
    return text


def canonicalize_invoice(obj: dict[str, Any]) -> dict[str, Any]:
    invoice = obj.get("invoice", {}) if isinstance(obj, dict) else {}
    items = obj.get("items", []) if isinstance(obj, dict) else []

    normalized_invoice = {
        "client_name": _clean_text(invoice.get("client_name")),
        "client_address": _clean_text(invoice.get("client_address")),
        "seller_name": _clean_text(invoice.get("seller_name")),
        "seller_address": _clean_text(invoice.get("seller_address")),
        "invoice_number": _clean_text(invoice.get("invoice_number")),
        "invoice_date": _normalize_date(invoice.get("invoice_date")),
        "due_date": _normalize_date(invoice.get("due_date")),
    }

    normalized_items: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        normalized_items.append(
            {
                "description": _clean_text(item.get("description")),
                "quantity": _clean_number(item.get("quantity")),
                "total_price": _clean_number(item.get("total_price")),
            }
        )

    return {"invoice": normalized_invoice, "items": normalized_items}

