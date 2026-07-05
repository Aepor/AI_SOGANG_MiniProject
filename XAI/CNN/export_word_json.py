"""Export CNN XAI scores as eojeol-level JSON files.

This script is intentionally an output adapter around the existing CNN XAI
method modules. The model still receives the original Okt-tokenized input, but
the exported scores are grouped back to whitespace-separated Korean eojeols so
the CNN/FNN/Transformer outputs can be compared in one presentation-friendly
shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.CNN.shared.paths import (  # noqa: E402
    default_cnn_output_dir,
    default_cnn_cache_path,
    default_cnn_model_path,
)
from XAI.CNN.shared.schemas import LABEL_NAMES, SampleRecord, make_sample_record  # noqa: E402
from XAI.CNN.shared.tokenization import encode_tokens, make_okt, tokenize_text  # noqa: E402
from XAI.CNN.xai_methods.filter_activation import run_filter_activation_analysis  # noqa: E402
from XAI.CNN.xai_methods.integrated_gradients import run_integrated_gradients  # noqa: E402
from XAI.CNN.xai_methods.loader import build_or_load_vocab_cache, load_cnn_model  # noqa: E402
from XAI.CNN.xai_methods.model import CNN_Sentiment, predict_one  # noqa: E402
from XAI.CNN.xai_methods.ngram_occlusion import run_ngram_occlusion  # noqa: E402
from XAI.CNN.xai_methods.unigram_occlusion import run_unigram_occlusion  # noqa: E402


POSITIVE_CLASS = 1
METHOD_NAMES = {
    "unigram_occlusion",
    "ngram_occlusion",
    "filter_activation",
    "integrated_gradients",
}
METHOD_ALIASES = {
    "integrated_gradient": "integrated_gradients",
}


def choose_device(requested: str) -> torch.device:
    """Resolve a CLI device option."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integer CLI values."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_positive_int_list(value: str, option_name: str) -> list[int]:
    """Parse positive comma-separated integer CLI values without duplicates."""
    values = list(dict.fromkeys(parse_int_list(value)))
    if not values:
        raise ValueError(f"{option_name} must include at least one integer.")
    invalid = [value for value in values if value <= 0]
    if invalid:
        invalid_text = ", ".join(str(value) for value in invalid)
        raise ValueError(f"{option_name} must be positive. Invalid values: {invalid_text}")
    return values


def normalize_method_name(method: str) -> str:
    """Map user-facing aliases to the internal method name."""
    return METHOD_ALIASES.get(method, method)


def read_input_texts(path: Path) -> list[str]:
    """Read one sentence per line from a UTF-8 text file."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip()]


def l1_normalize(scores: Iterable[float]) -> list[float]:
    """Normalize signed scores so sum(abs(score)) == 1 when possible."""
    values = [float(score) for score in scores]
    denom = sum(abs(score) for score in values)
    if denom == 0.0:
        return [0.0 for _score in values]
    return [score / denom for score in values]


def format_scores(scores: Iterable[float], digits: int | None) -> list[float]:
    """Return JSON-ready scores, optionally rounded for compact output."""
    if digits is None:
        return [float(score) for score in scores]
    return [round(float(score), digits) for score in scores]


def build_eojeol_spans(
    text: str,
    model_tokens: list[str],
    okt: Any,
    max_len: int,
) -> tuple[list[str], list[list[int]]]:
    """Map model token positions back to whitespace-separated eojeols.

    The CNN model is trained on Okt morph tokens after stopword removal. For
    display, each eojeol receives the sum of the model-token scores that came
    from tokenizing that eojeol. Eojeols removed entirely by the stopword filter
    keep an empty span and therefore receive a 0 score.
    """
    words = text.split()
    spans: list[list[int]] = []
    cursor = 0
    capped_tokens = model_tokens[:max_len]

    for word in words:
        word_tokens = tokenize_text(word, okt)[: max(0, max_len - cursor)]
        expected_end = min(cursor + len(word_tokens), len(capped_tokens))

        if word_tokens and capped_tokens[cursor:expected_end] == word_tokens[: expected_end - cursor]:
            span = list(range(cursor, expected_end))
            cursor = expected_end
        elif not word_tokens:
            span = []
        else:
            span = []
            for token in word_tokens:
                if cursor >= len(capped_tokens):
                    break
                if capped_tokens[cursor] == token:
                    span.append(cursor)
                    cursor += 1
                else:
                    try:
                        found_at = capped_tokens.index(token, cursor)
                    except ValueError:
                        continue
                    span.extend(range(cursor, found_at + 1))
                    cursor = found_at + 1
        spans.append(span)

    return words, spans


def token_scores_to_word_scores(token_scores: list[float], spans: list[list[int]]) -> list[float]:
    """Sum token-level scores into eojeol-level scores."""
    word_scores: list[float] = []
    for span in spans:
        word_scores.append(sum(token_scores[pos] for pos in span if pos < len(token_scores)))
    return l1_normalize(word_scores)


def rows_for_sample(rows: list[dict[str, Any]], sample_id: str) -> list[dict[str, Any]]:
    """Filter method rows for one sample."""
    return [row for row in rows if row.get("sample_id") == sample_id]


def unigram_word_scores(
    rows: list[dict[str, Any]], record: SampleRecord, spans: list[list[int]]
) -> list[float]:
    """Convert positive-class unigram occlusion rows to normalized eojeol scores."""
    token_scores = [0.0] * record.original_len
    for row in rows_for_sample(rows, record.sample_id):
        position = int(row["position"])
        if position < len(token_scores):
            token_scores[position] = float(row["prob_drop"])
    return token_scores_to_word_scores(token_scores, spans)


def integrated_gradients_word_scores(
    rows: list[dict[str, Any]], record: SampleRecord, spans: list[list[int]]
) -> list[float]:
    """Convert positive-class IG rows to normalized eojeol scores."""
    token_scores = [0.0] * record.original_len
    for row in rows_for_sample(rows, record.sample_id):
        position = int(row["position"])
        if position < len(token_scores):
            token_scores[position] = float(row["signed_score"])
    return token_scores_to_word_scores(token_scores, spans)


def ngram_word_scores(
    rows: list[dict[str, Any]], record: SampleRecord, spans: list[list[int]]
) -> list[float]:
    """Project n-gram occlusion windows to normalized eojeol scores.

    A window score is divided evenly across the model tokens inside that window.
    This keeps the final one-number-per-eojeol output comparable with unigram
    occlusion while still reflecting wider CNN n-gram evidence.
    """
    token_scores = [0.0] * record.original_len
    for row in rows_for_sample(rows, record.sample_id):
        start = int(row["start_pos"])
        end = int(row["end_pos"])
        width = max(1, end - start)
        share = float(row["prob_drop"]) / width
        for position in range(start, min(end, len(token_scores))):
            token_scores[position] += share
    return token_scores_to_word_scores(token_scores, spans)


def filter_activation_word_scores(
    rows: list[dict[str, Any]], record: SampleRecord, spans: list[list[int]]
) -> list[float]:
    """Project positive-vs-negative filter contributions to eojeol scores.

    Each selected filter n-gram contribution is divided evenly across the tokens
    in that activated window before eojeol aggregation. The score uses
    ``activation * (positive_fc_weight - negative_fc_weight)`` so positive
    values mean evidence toward the positive logit margin and negative values
    mean evidence toward the negative logit margin.
    """
    token_scores = [0.0] * record.original_len
    for row in rows_for_sample(rows, record.sample_id):
        start = int(row["position"])
        width = int(row["filter_size"])
        contribution = row.get("positive_margin_contribution")
        if contribution is None:
            contribution = float(row["activation"]) * float(row["positive_direction"])
        share = float(contribution) / max(1, width)
        for position in range(start, min(start + width, len(token_scores))):
            token_scores[position] += share
    return token_scores_to_word_scores(token_scores, spans)


def build_records(
    texts: list[str],
    model: CNN_Sentiment,
    word_to_index: dict[str, int],
    max_len: int,
    device: torch.device,
    batch_size: int,
    okt: Any,
) -> tuple[list[SampleRecord], dict[str, dict[str, Any]]]:
    """Build positive-class SampleRecords and per-sample display metadata."""
    records: list[SampleRecord] = []
    display: dict[str, dict[str, Any]] = {}

    for idx, text in enumerate(texts, start=1):
        tokens = tokenize_text(text, okt)
        ids, original_len = encode_tokens(tokens, word_to_index, max_len)
        pred_info = predict_one(model, ids, device, batch_size)
        record = make_sample_record(
            sample_id=f"input_{idx}",
            source="input_file",
            original_index=str(idx),
            text=text,
            true_label=None,
            tokens=tokens[:max_len],
            ids=ids,
            original_len=original_len,
            pred_info=pred_info,
            target_class=POSITIVE_CLASS,
        )
        words, spans = build_eojeol_spans(text, record.tokens, okt, max_len)
        records.append(record)
        display[record.sample_id] = {"words": words, "spans": spans}

    return records, display


def base_json_item(record: SampleRecord, words: list[str]) -> dict[str, Any]:
    """Create method-independent JSON fields."""
    return {
        "text": record.text,
        "prediction": LABEL_NAMES[record.pred_label],
        "probability": float(record.confidence),
        "words": words,
    }


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSON with readable Korean text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def method_payload(
    records: list[SampleRecord],
    display: dict[str, dict[str, Any]],
    method: str,
    rows: list[dict[str, Any]],
    score_digits: int | None,
) -> list[dict[str, Any]]:
    """Build the final list of JSON objects for one method."""
    payload: list[dict[str, Any]] = []

    for record in records:
        words = display[record.sample_id]["words"]
        spans = display[record.sample_id]["spans"]
        if method == "unigram_occlusion":
            scores = unigram_word_scores(rows, record, spans)
        elif method == "ngram_occlusion":
            scores = ngram_word_scores(rows, record, spans)
        elif method == "filter_activation":
            scores = filter_activation_word_scores(rows, record, spans)
        elif method == "integrated_gradients":
            scores = integrated_gradients_word_scores(rows, record, spans)
        else:
            raise ValueError(f"Unknown method: {method}")

        item = base_json_item(record, words)
        item["scores"] = format_scores(scores, score_digits)
        payload.append(item)

    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Export CNN XAI results from inputs.txt to eojeol-level JSON."
    )
    parser.add_argument("--input-file", type=Path, default=Path("inputs.txt"))
    parser.add_argument("--output-dir", type=Path, default=default_cnn_output_dir())
    parser.add_argument("--model-path", type=Path, default=default_cnn_model_path())
    parser.add_argument("--cache-path", type=Path, default=default_cnn_cache_path())
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ngram-sizes", default="2,3,4,5")
    parser.add_argument(
        "--ig-steps",
        default="50,100",
        help="Comma-separated Integrated Gradients step counts to export.",
    )
    parser.add_argument(
        "--score-digits",
        type=int,
        default=None,
        help="Optionally round output scores. By default full float values are written.",
    )
    parser.add_argument(
        "--methods",
        default="unigram_occlusion,ngram_occlusion,filter_activation,integrated_gradients",
        help="Comma-separated methods to export.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the export."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    texts = read_input_texts(args.input_file)
    if not texts:
        raise ValueError(f"No input sentences found in {args.input_file}")

    device = choose_device(args.device)
    cache = build_or_load_vocab_cache(args.cache_path, args.refresh_cache)
    word_to_index = cache["word_to_index"]
    vocab = cache["vocab"]
    model, _arch = load_cnn_model(args.model_path, len(vocab), device)

    okt = make_okt()
    records, display = build_records(
        texts=texts,
        model=model,
        word_to_index=word_to_index,
        max_len=args.max_len,
        device=device,
        batch_size=args.batch_size,
        okt=okt,
    )

    pad_idx = word_to_index.get("<pad>", 0)
    requested_methods_raw = [
        normalize_method_name(method.strip())
        for method in args.methods.split(",")
        if method.strip()
    ]
    requested_methods = list(dict.fromkeys(requested_methods_raw))
    unknown = sorted(set(requested_methods) - METHOD_NAMES)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    ig_steps_list = (
        parse_positive_int_list(args.ig_steps, "--ig-steps")
        if "integrated_gradients" in requested_methods
        else []
    )

    method_rows: dict[str, list[dict[str, Any]]] = {}
    integrated_gradients_rows_by_step: dict[int, list[dict[str, Any]]] = {}
    if "unigram_occlusion" in requested_methods:
        method_rows["unigram_occlusion"] = run_unigram_occlusion(
            model=model,
            samples=records,
            pad_idx=pad_idx,
            device=device,
            batch_size=args.batch_size,
        )
    if "ngram_occlusion" in requested_methods:
        method_rows["ngram_occlusion"] = run_ngram_occlusion(
            model=model,
            samples=records,
            pad_idx=pad_idx,
            ngram_sizes=parse_int_list(args.ngram_sizes),
            device=device,
            batch_size=args.batch_size,
        )
    if "filter_activation" in requested_methods:
        method_rows["filter_activation"] = run_filter_activation_analysis(
            model=model,
            samples=records,
            device=device,
        )
    if "integrated_gradients" in requested_methods:
        for ig_steps in ig_steps_list:
            integrated_gradients_rows_by_step[ig_steps] = run_integrated_gradients(
                model=model,
                samples=records,
                pad_idx=pad_idx,
                device=device,
                steps=ig_steps,
            )

    for method in requested_methods:
        if method == "integrated_gradients":
            for idx, (ig_steps, rows) in enumerate(integrated_gradients_rows_by_step.items()):
                payload = method_payload(
                    records=records,
                    display=display,
                    method=method,
                    rows=rows,
                    score_digits=args.score_digits,
                )
                output_path = args.output_dir / f"output_cnn_{method}_steps{ig_steps}.json"
                write_json(output_path, payload)
                print(f"Wrote {output_path} ({len(payload)} rows)")

                if idx == 0:
                    compatibility_path = args.output_dir / f"output_cnn_{method}.json"
                    write_json(compatibility_path, payload)
                    print(f"Wrote {compatibility_path} ({len(payload)} rows)")
            continue

        payload = method_payload(
            records=records,
            display=display,
            method=method,
            rows=method_rows[method],
            score_digits=args.score_digits,
        )
        output_path = args.output_dir / f"output_cnn_{method}.json"
        write_json(output_path, payload)
        print(f"Wrote {output_path} ({len(payload)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
