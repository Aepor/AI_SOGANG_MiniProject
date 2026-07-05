"""Compare Opus evidence vectors with unified XAI word-score vectors."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.OpusBaseline.common import (  # noqa: E402
    comparison_dir,
    cosine_similarity,
    read_jsonl,
    sign,
    topk_abs_indices,
    validate_words_scores,
    write_csv,
)


SCORE_FIELDS = [
    "sample_id",
    "model",
    "method",
    "prediction",
    "probability",
    "opus_sentiment",
    "prediction_opus_agreement",
    "topk_recall",
    "jaccard",
    "signed_cosine",
    "evidence_mass",
    "polarity_agreement",
    "opus_match_score",
    "xai_top_words",
    "opus_evidence_words",
    "sample_type",
]


def opus_match_score(
    topk_recall: float,
    jaccard: float,
    signed_cosine: float,
    evidence_mass: float,
) -> float:
    """Composite presentation score."""
    return (
        0.35 * topk_recall
        + 0.25 * jaccard
        + 0.25 * max(0.0, signed_cosine)
        + 0.15 * evidence_mass
    )


def compare_one(xai: dict[str, Any], opus: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Compare one XAI row against one Opus vector row."""
    sample_id = str(xai["sample_id"])
    words = list(xai["words"])
    xai_scores = [float(value) for value in xai["scores"]]
    opus_vector = [float(value) for value in opus["opus_vector"]]
    opus_indices = set(int(idx) for idx in opus.get("opus_evidence_indices", []))
    validate_words_scores(sample_id, words, xai_scores)
    validate_words_scores(sample_id, list(opus["words"]), opus_vector)
    if words != list(opus["words"]):
        raise ValueError(f"{sample_id}: XAI words and Opus words differ")

    k = min(top_k, len(words))
    xai_top = set(topk_abs_indices(xai_scores, k))
    intersection = xai_top & opus_indices
    union = xai_top | opus_indices
    topk_recall = 0.0 if not opus_indices else len(intersection) / len(opus_indices)
    jaccard = 0.0 if not union else len(intersection) / len(union)
    evidence_mass = sum(abs(xai_scores[idx]) for idx in opus_indices)
    signed_cosine = cosine_similarity(opus_vector, xai_scores)

    same_sign = 0
    comparable = 0
    for idx in intersection:
        opus_sign = sign(opus_vector[idx])
        xai_sign = sign(xai_scores[idx])
        if opus_sign == 0 or xai_sign == 0:
            continue
        comparable += 1
        if opus_sign == xai_sign:
            same_sign += 1
    polarity_agreement = 0.0 if comparable == 0 else same_sign / comparable
    match = opus_match_score(topk_recall, jaccard, signed_cosine, evidence_mass)

    prediction = str(xai.get("prediction", ""))
    opus_sentiment = str(opus["opus_sentiment"])
    return {
        "sample_id": sample_id,
        "model": xai["model"],
        "method": xai["method"],
        "prediction": prediction,
        "probability": xai.get("probability", ""),
        "opus_sentiment": opus_sentiment,
        "prediction_opus_agreement": prediction == opus_sentiment,
        "topk_recall": round(topk_recall, 6),
        "jaccard": round(jaccard, 6),
        "signed_cosine": round(signed_cosine, 6),
        "evidence_mass": round(evidence_mass, 6),
        "polarity_agreement": round(polarity_agreement, 6),
        "opus_match_score": round(match, 6),
        "xai_top_words": " | ".join(words[idx] for idx in sorted(xai_top)),
        "opus_evidence_words": " | ".join(words[idx] for idx in sorted(opus_indices)),
        "sample_type": xai.get("sample_type", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize metrics by model and method."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]))].append(row)

    summary: list[dict[str, Any]] = []
    metric_names = [
        "topk_recall",
        "jaccard",
        "signed_cosine",
        "evidence_mass",
        "polarity_agreement",
        "opus_match_score",
    ]
    for (model, method), group_rows in grouped.items():
        item: dict[str, Any] = {"model": model, "method": method, "n_samples": len(group_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows]
            item[f"mean_{metric}"] = round(statistics.mean(values), 6)
        item["prediction_opus_agreement_rate"] = round(
            statistics.mean(1.0 if row["prediction_opus_agreement"] else 0.0 for row in group_rows),
            6,
        )
        summary.append(item)

    summary.sort(key=lambda row: float(row["mean_opus_match_score"]), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def write_case_report(path: Path, score_rows: list[dict[str, Any]], opus_by_id: dict[str, dict[str, Any]]) -> None:
    """Write a compact qualitative markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_sample[str(row["sample_id"])].append(row)

    lines = ["# Opus-XAI Qualitative Case Report", ""]
    for sample_id in sorted(by_sample):
        opus = opus_by_id[sample_id]
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"text: {opus.get('text', '')}",
                "",
                f"- Opus sentiment: {opus['opus_sentiment']}",
                f"- Opus evidence: {', '.join(str(e['phrase']) for e in opus.get('evidence', []))}",
                f"- Opus reason: {opus.get('brief_reason', '')}",
                "",
                "| model | method | prediction | top words | score |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for row in sorted(by_sample[sample_id], key=lambda item: float(item["opus_match_score"]), reverse=True):
            lines.append(
                f"| {row['model']} | {row['method']} | {row['prediction']} | "
                f"{row['xai_top_words']} | {row['opus_match_score']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare Opus vectors with unified XAI outputs.")
    parser.add_argument("--xai", type=Path, default=comparison_dir() / "xai_unified.jsonl")
    parser.add_argument("--opus", type=Path, default=comparison_dir() / "opus_vectors.jsonl")
    parser.add_argument("--scores-output", type=Path, default=comparison_dir() / "opus_xai_overlap_scores.csv")
    parser.add_argument("--summary-output", type=Path, default=comparison_dir() / "opus_xai_method_summary.csv")
    parser.add_argument("--report-output", type=Path, default=comparison_dir() / "qualitative_case_report.md")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    xai_rows = read_jsonl(args.xai)
    opus_rows = read_jsonl(args.opus)
    opus_by_id = {str(row["sample_id"]): row for row in opus_rows}
    score_rows = [
        compare_one(row, opus_by_id[str(row["sample_id"])], args.top_k)
        for row in xai_rows
        if str(row["sample_id"]) in opus_by_id
    ]
    summary_rows = summarize(score_rows)
    write_csv(args.scores_output, score_rows, SCORE_FIELDS)
    summary_fields = [
        "model",
        "method",
        "n_samples",
        "mean_topk_recall",
        "mean_jaccard",
        "mean_signed_cosine",
        "mean_evidence_mass",
        "mean_polarity_agreement",
        "mean_opus_match_score",
        "prediction_opus_agreement_rate",
        "rank",
    ]
    write_csv(args.summary_output, summary_rows, summary_fields)
    write_case_report(args.report_output, score_rows, opus_by_id)
    print(f"Wrote {args.scores_output} ({len(score_rows)} rows)")
    print(f"Wrote {args.summary_output} ({len(summary_rows)} rows)")
    print(f"Wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
