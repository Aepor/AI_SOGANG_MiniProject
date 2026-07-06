"""Shared utilities for LLM-XAI comparison scripts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


POSITIVE = "positive"
NEGATIVE = "negative"
VALID_SENTIMENTS = {POSITIVE, NEGATIVE}


def repo_root() -> Path:
    """Return the repository root from this file location."""
    return Path(__file__).resolve().parents[2]


def xai_root() -> Path:
    """Return the XAI directory."""
    return repo_root() / "XAI"


def comparison_dir() -> Path:
    """Return the default comparison output directory."""
    return xai_root() / "LLMComparison" / "comparison_outputs"


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without overriding existing env vars."""
    env_path = repo_root() / ".env" if path is None else path
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for line_no, raw_line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line at {env_path}:{line_no}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env key at {env_path}:{line_no}")
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_lines(path: Path) -> list[str]:
    """Read non-empty UTF-8 lines."""
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as JSON Lines."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV with UTF-8 BOM for Excel compatibility."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_selected_reviews(path: Path) -> list[dict[str, str]]:
    """Read selected review rows from CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader if row.get("sample_id") and row.get("text")]


def build_selected_reviews_from_texts(texts: list[str]) -> list[dict[str, str]]:
    """Build selected review rows from plain text lines."""
    return [
        {
            "sample_id": f"case_{idx:03d}",
            "text": text,
            "true_label": "",
            "sample_type": "custom",
            "selection_reason": "inputs.txt",
        }
        for idx, text in enumerate(texts, start=1)
    ]


def write_selected_reviews(path: Path, rows: list[dict[str, str]]) -> None:
    """Write selected review CSV."""
    write_csv(
        path,
        rows,
        ["sample_id", "text", "true_label", "sample_type", "selection_reason"],
    )


def l1_normalize(values: Iterable[float]) -> list[float]:
    """Normalize a signed vector so sum(abs(x)) == 1 when possible."""
    vector = [float(value) for value in values]
    denom = sum(abs(value) for value in vector)
    if denom == 0.0:
        return [0.0 for _value in vector]
    return [value / denom for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for equal-length vectors."""
    if len(left) != len(right):
        raise ValueError(f"Vector length mismatch: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def topk_abs_indices(values: list[float], k: int) -> list[int]:
    """Return indices of the k largest absolute values."""
    if k <= 0:
        return []
    return [
        idx
        for idx, _value in sorted(
            enumerate(values),
            key=lambda pair: (abs(pair[1]), -pair[0]),
            reverse=True,
        )[:k]
    ]


def sign(value: float) -> int:
    """Return the sign of a numeric value."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def sentiment_to_sign(sentiment: str) -> int:
    """Map sentiment labels to score signs."""
    if sentiment == POSITIVE:
        return 1
    if sentiment == NEGATIVE:
        return -1
    raise ValueError(f"Invalid sentiment/polarity: {sentiment}")


def validate_words_scores(sample_id: str, words: list[Any], scores: list[Any]) -> None:
    """Validate that words and score arrays align."""
    if len(words) != len(scores):
        raise ValueError(
            f"{sample_id}: words/scores length mismatch: {len(words)} != {len(scores)}"
        )
