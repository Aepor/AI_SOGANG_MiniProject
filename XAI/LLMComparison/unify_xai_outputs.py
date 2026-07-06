"""Convert model-specific XAI JSON files into one comparison JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.LLMComparison.common import (  # noqa: E402
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
    "integrated_gradients_steps50": "output_cnn_integrated_gradients_steps50.json",
    "integrated_gradients_steps100": "output_cnn_integrated_gradients_steps100.json",
}

FNN_METHOD_FILES = {
    "occlusion": "output_fnn_occlusion.json",
    "lime": "output_fnn_lime.json",
    "integrated_gradients_steps50": "output_fnn_ig_50.json",
    "integrated_gradients_steps100": "output_fnn_ig_100.json",
}

TRANSFORMER_METHOD_FILES = {
    "occlusion": "output_transformer_occlusion.json",
    "integrated_gradients_steps50": "output_transformer_ig_50.json",
    "integrated_gradients_steps100": "output_transformer_ig_100.json",
}


def l1_abs_sum(scores: list[float]) -> float:
    """Return sum(abs(score)) for a score vector."""
    return sum(abs(score) for score in scores)


def is_l1_normalized(scores: list[float], tolerance: float) -> bool:
    """Return whether scores are already L1-normalized, allowing zero vectors."""
    total = l1_abs_sum(scores)
    return total == 0.0 or abs(total - 1.0) <= tolerance


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


def unify_model_outputs(
    model: str,
    method_files: dict[str, str],
    output_dir: Path,
    selected_reviews: Path,
    renormalize_scores: bool,
    l1_tolerance: float,
) -> list[dict[str, Any]]:
    """Unify model method JSON files that share the exported word-score schema."""
    _by_id, by_text = selected_lookup(selected_reviews)
    unified: list[dict[str, Any]] = []

    for method, filename in method_files.items():
        path = output_dir / filename
        if not path.exists():
            print(f"Skipping missing {model.upper()} output: {path}")
            continue
        not_l1_count = 0
        unmatched_count = 0
        for index, item in enumerate(load_json_array(path), start=1):
            text = str(item["text"])
            sample = by_text.get(text)
            if sample is None:
                unmatched_count += 1
                continue
            sample_id = str(sample["sample_id"])
            source_words = list(item["words"])
            selected_words = str(sample["text"]).split() if sample is not None else []
            words_aligned_to_selected_text = False
            words = source_words
            if selected_words and selected_words != source_words and len(selected_words) == len(source_words):
                words = selected_words
                words_aligned_to_selected_text = True
            score_key = find_score_key(item)
            scores = [float(value) for value in item[score_key]]
            validate_words_scores(sample_id, words, scores)
            score_l1_sum = l1_abs_sum(scores)
            l1_normalized = is_l1_normalized(scores, l1_tolerance)
            if not l1_normalized:
                not_l1_count += 1

            normalize_for_comparison = renormalize_scores or not l1_normalized
            output_scores = l1_normalize(scores) if normalize_for_comparison else scores
            unified.append(
                {
                    "sample_id": sample_id,
                    "model": model,
                    "method": method,
                    "text": text,
                    "prediction": item.get("prediction", ""),
                    "probability": item.get("probability", ""),
                    "words": words,
                    "scores": output_scores,
                    "score_key": score_key,
                    "score_target": "positive",
                    "score_normalization": (
                        "word_l1_renormalized"
                        if normalize_for_comparison
                        else "word_l1_exported"
                    ),
                    "source_words": source_words,
                    "words_aligned_to_selected_text": words_aligned_to_selected_text,
                    "source_score_l1_sum": round(score_l1_sum, 12),
                    "source_scores_l1_normalized": l1_normalized,
                    "true_label": "" if sample is None else sample.get("true_label", ""),
                    "sample_type": "" if sample is None else sample.get("sample_type", ""),
                    "source_file": str(path.as_posix()),
                }
            )
        if not_l1_count:
            print(
                f"Info: {path} has {not_l1_count} rows whose source scores are not "
                f"L1-normalized within tolerance {l1_tolerance}; normalized once for comparison."
            )
        if unmatched_count:
            print(
                f"Warning: skipped {unmatched_count} {model.upper()} rows from {path} because "
                "their text was not found in selected_reviews.csv."
            )
    return unified


def unify_cnn_outputs(
    output_dir: Path,
    selected_reviews: Path,
    renormalize_scores: bool,
    l1_tolerance: float,
) -> list[dict[str, Any]]:
    """Unify CNN method JSON files."""
    return unify_model_outputs(
        "cnn",
        CNN_METHOD_FILES,
        output_dir,
        selected_reviews,
        renormalize_scores,
        l1_tolerance,
    )


def unify_fnn_outputs(
    output_dir: Path,
    selected_reviews: Path,
    renormalize_scores: bool,
    l1_tolerance: float,
) -> list[dict[str, Any]]:
    """Unify FNN method JSON files."""
    return unify_model_outputs(
        "fnn",
        FNN_METHOD_FILES,
        output_dir,
        selected_reviews,
        renormalize_scores,
        l1_tolerance,
    )


def unify_transformer_outputs(
    output_dir: Path,
    selected_reviews: Path,
    renormalize_scores: bool,
    l1_tolerance: float,
) -> list[dict[str, Any]]:
    """Unify Transformer method JSON files."""
    return unify_model_outputs(
        "transformer",
        TRANSFORMER_METHOD_FILES,
        output_dir,
        selected_reviews,
        renormalize_scores,
        l1_tolerance,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Unify model XAI outputs for LLM comparison.")
    parser.add_argument(
        "--selected-reviews",
        type=Path,
        default=comparison_dir() / "selected_reviews.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=xai_root() / "outputs_json")
    parser.add_argument("--cnn-output-dir", type=Path, default=None)
    parser.add_argument("--fnn-output-dir", type=Path, default=None)
    parser.add_argument("--transformer-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=comparison_dir() / "xai_unified.jsonl")
    parser.add_argument("--models", default="cnn,fnn,transformer")
    parser.add_argument(
        "--renormalize-scores",
        action="store_true",
        help=(
            "Force L1 normalization while unifying. By default already-L1 scores are preserved "
            "and non-L1 scores are normalized once for comparison."
        ),
    )
    parser.add_argument("--l1-tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the unifier."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    models = {part.strip().lower() for part in args.models.split(",") if part.strip()}
    unified: list[dict[str, Any]] = []
    if "cnn" in models:
        unified.extend(
            unify_cnn_outputs(
                args.cnn_output_dir or args.output_dir,
                args.selected_reviews,
                args.renormalize_scores,
                args.l1_tolerance,
            )
        )
    if "fnn" in models:
        unified.extend(
            unify_fnn_outputs(
                args.fnn_output_dir or args.output_dir,
                args.selected_reviews,
                args.renormalize_scores,
                args.l1_tolerance,
            )
        )
    if "transformer" in models:
        unified.extend(
            unify_transformer_outputs(
                args.transformer_output_dir or args.output_dir,
                args.selected_reviews,
                args.renormalize_scores,
                args.l1_tolerance,
            )
        )
    unsupported = sorted(models - {"cnn", "fnn", "transformer"})
    if unsupported:
        print(f"Unsupported models for now, skipped: {', '.join(unsupported)}")

    write_jsonl(args.output, unified)
    print(f"Wrote {args.output} ({len(unified)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
