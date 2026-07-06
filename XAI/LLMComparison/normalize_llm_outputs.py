"""Validate LLM outputs and convert evidence to signed word vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.LLMComparison.common import (  # noqa: E402
    VALID_SENTIMENTS,
    comparison_dir,
    l1_normalize,
    read_jsonl,
    sentiment_to_sign,
    write_jsonl,
)


def validate_evidence_item(sample_id: str, item: dict[str, Any], word_count: int) -> dict[str, Any]:
    """Validate and normalize one evidence item."""
    phrase = str(item.get("phrase", "")).strip()
    polarity = str(item.get("polarity", "")).strip().lower()
    if not phrase:
        raise ValueError(f"{sample_id}: evidence phrase is empty")
    if polarity not in VALID_SENTIMENTS:
        raise ValueError(f"{sample_id}: invalid evidence polarity: {polarity}")
    indices = item.get("word_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{sample_id}: evidence word_indices must be a non-empty list")
    normalized_indices: list[int] = []
    for raw_idx in indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= word_count:
            raise ValueError(f"{sample_id}: word index out of range: {idx}")
        normalized_indices.append(idx)
    strength = float(item.get("strength", 1.0))
    if strength < 0.0:
        raise ValueError(f"{sample_id}: evidence strength must be non-negative")
    return {
        "phrase": phrase,
        "word_indices": sorted(dict.fromkeys(normalized_indices)),
        "polarity": polarity,
        "strength": strength,
        "reason": str(item.get("reason", "")).strip(),
    }


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one LLM explanation record."""
    sample_id = str(row.get("sample_id", "")).strip()
    sentiment = str(row.get("sentiment", "")).strip().lower()
    words = row.get("words")
    if not sample_id:
        raise ValueError("LLM row missing sample_id")
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"{sample_id}: invalid sentiment: {sentiment}")
    if not isinstance(words, list) or not words:
        raise ValueError(f"{sample_id}: words must be a non-empty list")

    evidence = row.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{sample_id}: evidence must be a non-empty list")
    normalized_evidence = [
        validate_evidence_item(sample_id, dict(item), len(words)) for item in evidence
    ]

    vector = [0.0] * len(words)
    evidence_indices: set[int] = set()
    for item in normalized_evidence:
        contribution = sentiment_to_sign(item["polarity"]) * float(item["strength"])
        for idx in item["word_indices"]:
            vector[idx] += contribution
            evidence_indices.add(idx)

    return {
        "sample_id": sample_id,
        "text": row.get("text", ""),
        "words": [str(word) for word in words],
        "llm_sentiment": sentiment,
        "llm_vector": l1_normalize(vector),
        "llm_evidence_indices": sorted(evidence_indices),
        "evidence": normalized_evidence,
        "brief_reason": str(row.get("brief_reason", "")).strip(),
        "model_id": row.get("model_id", ""),
        "prompt_version": row.get("prompt_version", ""),
        "created_at": row.get("created_at", ""),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Normalize LLM explanation JSONL.")
    parser.add_argument("--input", type=Path, default=comparison_dir() / "llm_explanations.jsonl")
    parser.add_argument("--output", type=Path, default=comparison_dir() / "llm_vectors.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run normalization."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = [normalize_record(row) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
