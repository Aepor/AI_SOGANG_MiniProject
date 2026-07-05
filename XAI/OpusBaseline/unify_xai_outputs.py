"""Convert model-specific XAI JSON files into one comparison JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.OpusBaseline.common import (  # noqa: E402
    comparison_dir,
    l1_normalize,
    read_selected_reviews,
    validate_words_scores,
    write_jsonl,
    xai_root,
)


CNN_METHOD_FILES = {
    "unigram_occlusion": "output_cnn_unigram_occlusion.json",
    "ngram_occlusion": "output_cnn_ngram_occlusion.json",
    "filter_activation": "output_cnn_filter_activation.json",
    "integrated_gradients": "output_cnn_integrated_gradients.json",
}


def find_score_key(item: dict[str, Any]) -> str:
    """Find the single score array key in an exported XAI item."""
    if "scores" in item:
        return "scores"
    keys = [key for key in item if key.endswith("_scores")]
    if len(keys) != 1:
        raise ValueError(f"Expected scores or exactly one *_scores key, found {keys}")
    return keys[0]


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array from disk."""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array in {path}")
    return value


def selected_lookup(selected_reviews: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Build sample lookup maps by sample_id and text."""
    rows = read_selected_reviews(selected_reviews)
    by_id = {row["sample_id"]: row for row in rows}
    by_text = {row["text"]: row for row in rows}
    return by_id, by_text


def unify_cnn_outputs(cnn_output_dir: Path, selected_reviews: Path) -> list[dict[str, Any]]:
    """Unify CNN method JSON files."""
    _by_id, by_text = selected_lookup(selected_reviews)
    unified: list[dict[str, Any]] = []

    for method, filename in CNN_METHOD_FILES.items():
        path = cnn_output_dir / filename
        if not path.exists():
            print(f"Skipping missing CNN output: {path}")
            continue
        for index, item in enumerate(load_json_array(path), start=1):
            text = str(item["text"])
            sample = by_text.get(text)
            sample_id = str((sample or {}).get("sample_id") or item.get("sample_id") or f"case_{index:03d}")
            words = list(item["words"])
            score_key = find_score_key(item)
            scores = [float(value) for value in item[score_key]]
            validate_words_scores(sample_id, words, scores)

            # Keep the exported signed direction, but make sure old outputs are L1-normalized.
            normalized_scores = l1_normalize(scores)
            unified.append(
                {
                    "sample_id": sample_id,
                    "model": "cnn",
                    "method": method,
                    "text": text,
                    "prediction": item.get("prediction", ""),
                    "probability": item.get("probability", ""),
                    "words": words,
                    "scores": normalized_scores,
                    "score_key": score_key,
                    "score_target": "positive",
                    "score_normalization": "word_l1",
                    "true_label": "" if sample is None else sample.get("true_label", ""),
                    "sample_type": "" if sample is None else sample.get("sample_type", ""),
                    "source_file": str(path.as_posix()),
                }
            )
    return unified


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Unify model XAI outputs for Opus comparison.")
    parser.add_argument(
        "--selected-reviews",
        type=Path,
        default=comparison_dir() / "selected_reviews.csv",
    )
    parser.add_argument("--cnn-output-dir", type=Path, default=xai_root() / "CNN" / "outputs")
    parser.add_argument("--output", type=Path, default=comparison_dir() / "xai_unified.jsonl")
    parser.add_argument("--models", default="cnn")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the unifier."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    models = {part.strip().lower() for part in args.models.split(",") if part.strip()}
    unified: list[dict[str, Any]] = []
    if "cnn" in models:
        unified.extend(unify_cnn_outputs(args.cnn_output_dir, args.selected_reviews))
    unsupported = sorted(models - {"cnn"})
    if unsupported:
        print(f"Unsupported models for now, skipped: {', '.join(unsupported)}")

    write_jsonl(args.output, unified)
    print(f"Wrote {args.output} ({len(unified)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
