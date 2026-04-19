from __future__ import annotations

import json
import os
import random
import re
import time
from urllib import error, request
from dataclasses import dataclass
from typing import Protocol

from .schema import canonicalize_invoice


class Extractor(Protocol):
    name: str

    def extract(self, ocr_text: str, prompt_variant: str, run_seed: int) -> dict:
        ...


@dataclass
class RegexBaselineExtractor:
    name: str = "regex_baseline"

    def extract(self, ocr_text: str, prompt_variant: str, run_seed: int) -> dict:
        # Deterministic baseline from OCR text only.
        inv = {
            "client_name": "",
            "client_address": "",
            "seller_name": "",
            "seller_address": "",
            "invoice_number": "",
            "invoice_date": "",
            "due_date": "",
        }
        items: list[dict[str, str]] = []
        patterns = {
            "invoice_number": r"(?:Invoice\s*(?:Number|#)\s*[:\-]?\s*)([A-Za-z0-9\-]+)",
            "invoice_date": r"(?:Invoice\s*Date\s*[:\-]?\s*)(\d{1,2}/\d{1,2}/\d{2,4})",
            "due_date": r"(?:Due\s*Date\s*[:\-]?\s*)(\d{1,2}/\d{1,2}/\d{2,4})",
        }
        for field, pattern in patterns.items():
            m = re.search(pattern, ocr_text, flags=re.IGNORECASE)
            if m:
                inv[field] = m.group(1).strip()

        # Light heuristic for line-items: "desc ... qty ... price"
        line_pattern = re.compile(
            r"([A-Za-z0-9 ,.'/\-]{8,}?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
        )
        for desc, qty, total in line_pattern.findall(ocr_text):
            if len(items) >= 8:
                break
            items.append(
                {"description": desc.strip(), "quantity": qty.strip(), "total_price": total.strip()}
            )
        return canonicalize_invoice({"invoice": inv, "items": items})


@dataclass
class SimulatedLLMExtractor:
    """Offline simulator to stress-test consistency without API dependencies."""

    name: str = "simulated_llm"

    def extract(self, ocr_text: str, prompt_variant: str, run_seed: int) -> dict:
        # Simulates an imperfect LLM by extracting from OCR text only.
        random_gen = random.Random(run_seed + hash(prompt_variant) % 10_000)
        parsed = self._extract_embedded_json(ocr_text)
        if not parsed:
            parsed = self._regex_seed_parse(ocr_text)
        if not parsed:
            parsed = {"invoice": {}, "items": []}
        parsed = canonicalize_invoice(parsed)

        error_rate = 0.04 if prompt_variant == "few_shot_schema" else 0.08
        # Drop or perturb fields to mimic extraction instability.
        for field in list(parsed.get("invoice", {})):
            if random_gen.random() < error_rate:
                if field in {"invoice_number", "invoice_date"}:
                    parsed["invoice"][field] = ""
                else:
                    parsed["invoice"][field] = parsed["invoice"][field][: max(0, len(parsed["invoice"][field]) - 2)]

        new_items = []
        for item in parsed.get("items", []):
            if random_gen.random() < error_rate / 2:
                continue
            new_item = dict(item)
            if random_gen.random() < error_rate:
                new_item["description"] = new_item["description"][:20]
            if random_gen.random() < error_rate:
                new_item["total_price"] = ""
            new_items.append(new_item)
        parsed["items"] = new_items
        return canonicalize_invoice(parsed)

    @staticmethod
    def _regex_seed_parse(ocr_text: str) -> dict:
        inv = {
            "client_name": "",
            "client_address": "",
            "seller_name": "",
            "seller_address": "",
            "invoice_number": "",
            "invoice_date": "",
            "due_date": "",
        }
        items: list[dict[str, str]] = []
        patterns = {
            "invoice_number": r"(?:Invoice\s*(?:Number|#)\s*[:\-]?\s*)([A-Za-z0-9\-]+)",
            "invoice_date": r"(?:Invoice\s*Date\s*[:\-]?\s*)(\d{1,2}/\d{1,2}/\d{2,4})",
            "due_date": r"(?:Due\s*Date\s*[:\-]?\s*)(\d{1,2}/\d{1,2}/\d{2,4})",
        }
        for field, pattern in patterns.items():
            m = re.search(pattern, ocr_text, flags=re.IGNORECASE)
            if m:
                inv[field] = m.group(1).strip()
        line_pattern = re.compile(
            r"([A-Za-z0-9 ,.'/\-]{8,}?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
        )
        for desc, qty, total in line_pattern.findall(ocr_text):
            if len(items) >= 8:
                break
            items.append(
                {"description": desc.strip(), "quantity": qty.strip(), "total_price": total.strip()}
            )
        return {"invoice": inv, "items": items}

    @staticmethod
    def _extract_embedded_json(text: str) -> dict | None:
        # This supports synthetic test prompts that include JSON payloads.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


@dataclass
class OpenAIGPT4oExtractor:
    """Production-ready GPT-4o extractor using OpenAI Responses API."""

    name: str = "openai_gpt4o"
    model: str = "gpt-4o"
    timeout_seconds: int = 60
    max_retries: int = 6
    temperature: float = 0.0
    fail_on_api_error: bool = False

    def extract(self, ocr_text: str, prompt_variant: str, run_seed: int) -> dict:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when production GPT-4o mode is enabled.")

        system_prompt = (
            "You extract structured invoice data from OCR text. "
            "Return only valid JSON with this exact schema: "
            "{\"invoice\":{\"client_name\":\"\",\"client_address\":\"\",\"seller_name\":\"\","
            "\"seller_address\":\"\",\"invoice_number\":\"\",\"invoice_date\":\"\",\"due_date\":\"\"},"
            "\"items\":[{\"description\":\"\",\"quantity\":\"\",\"total_price\":\"\"}]}"
        )
        prompt_style = (
            "Use strict key coverage and empty strings for missing values."
            if prompt_variant == "strict_json"
            else "Use schema-consistent extraction with robust normalization; still output only JSON."
        )
        user_prompt = (
            f"{prompt_style}\n\nOCR TEXT:\n{ocr_text}\n\n"
            "Output only JSON. No markdown, no explanation."
        )

        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "temperature": self.temperature,
            "max_output_tokens": 1200,
        }

        last_error: Exception | None = None
        rng = random.Random(run_seed + 113)
        for attempt in range(self.max_retries):
            try:
                text = self._call_openai(payload, api_key=api_key)
                parsed = self._extract_json_payload(text)
                if parsed is None:
                    return canonicalize_invoice({"invoice": {}, "items": []})
                return canonicalize_invoice(parsed)
            except error.HTTPError as exc:
                last_error = exc
                if exc.code == 429 and attempt < self.max_retries - 1:
                    retry_after = self._retry_after_seconds(exc)
                    sleep_s = retry_after if retry_after is not None else min(45.0, (2 ** attempt) + rng.random())
                    time.sleep(sleep_s)
                    continue
                if attempt < self.max_retries - 1:
                    time.sleep(min(30.0, (2 ** attempt) + rng.random()))
                    continue
                if self.fail_on_api_error:
                    raise RuntimeError(f"GPT-4o extraction failed after retries: {exc}") from exc
                return canonicalize_invoice({"invoice": {}, "items": []})
            except Exception as exc:  # network/parse/server side issues
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(30.0, (2 ** attempt) + rng.random()))
                    continue
                if self.fail_on_api_error:
                    raise RuntimeError(f"GPT-4o extraction failed after retries: {exc}") from exc
                return canonicalize_invoice({"invoice": {}, "items": []})
        if last_error is not None and self.fail_on_api_error:
            raise RuntimeError(f"GPT-4o extraction failed: {last_error}") from last_error
        return canonicalize_invoice({"invoice": {}, "items": []})

    def _call_openai(self, payload: dict, api_key: str) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url="https://api.openai.com/v1/responses",
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        output_text = data.get("output_text")
        if output_text:
            return str(output_text)
        # Fallback parsing for API shape variance.
        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
        raise error.HTTPError(
            url="https://api.openai.com/v1/responses",
            code=500,
            msg="No textual output returned by model.",
            hdrs=None,
            fp=None,
        )

    @staticmethod
    def _extract_json_payload(text: str) -> dict | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _retry_after_seconds(exc: error.HTTPError) -> float | None:
        header = None
        if exc.headers:
            header = exc.headers.get("Retry-After")
        if not header:
            return None
        try:
            return max(0.0, float(header))
        except ValueError:
            return None

