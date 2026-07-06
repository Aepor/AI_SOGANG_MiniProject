"""Compare LLM evidence vectors with unified XAI word-score vectors."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.LLMComparison.common import (  # noqa: E402
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
    "llm_sentiment",
    "prediction_llm_agreement",
    "topk_recall",
    "jaccard",
    "signed_cosine",
    "evidence_mass",
    "polarity_agreement",
    "llm_match_score",
    "xai_top_words",
    "llm_evidence_words",
    "sample_type",
]


SUMMARY_FIELDS = [
    "model",
    "method",
    "n_samples",
    "mean_topk_recall",
    "mean_jaccard",
    "mean_signed_cosine",
    "mean_evidence_mass",
    "mean_polarity_agreement",
    "mean_llm_match_score",
    "prediction_llm_agreement_rate",
    "rank",
]


def llm_match_score(
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


def compare_one(xai: dict[str, Any], llm: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Compare one XAI row against one LLM vector row."""
    sample_id = str(xai["sample_id"])
    words = list(xai["words"])
    xai_scores = [float(value) for value in xai["scores"]]
    llm_vector = [float(value) for value in llm["llm_vector"]]
    llm_indices = set(int(idx) for idx in llm.get("llm_evidence_indices", []))
    validate_words_scores(sample_id, words, xai_scores)
    validate_words_scores(sample_id, list(llm["words"]), llm_vector)
    if words != list(llm["words"]):
        raise ValueError(f"{sample_id}: XAI words and LLM words differ")

    k = min(top_k, len(words))
    xai_top = set(topk_abs_indices(xai_scores, k))
    intersection = xai_top & llm_indices
    union = xai_top | llm_indices
    topk_recall = 0.0 if not llm_indices else len(intersection) / len(llm_indices)
    jaccard = 0.0 if not union else len(intersection) / len(union)
    evidence_mass = sum(abs(xai_scores[idx]) for idx in llm_indices)
    signed_cosine = cosine_similarity(llm_vector, xai_scores)

    same_sign = 0
    comparable = 0
    for idx in intersection:
        llm_sign = sign(llm_vector[idx])
        xai_sign = sign(xai_scores[idx])
        if llm_sign == 0 or xai_sign == 0:
            continue
        comparable += 1
        if llm_sign == xai_sign:
            same_sign += 1
    polarity_agreement = 0.0 if comparable == 0 else same_sign / comparable
    match = llm_match_score(topk_recall, jaccard, signed_cosine, evidence_mass)

    prediction = str(xai.get("prediction", ""))
    llm_sentiment = str(llm["llm_sentiment"])
    return {
        "sample_id": sample_id,
        "model": xai["model"],
        "method": xai["method"],
        "prediction": prediction,
        "probability": xai.get("probability", ""),
        "llm_sentiment": llm_sentiment,
        "prediction_llm_agreement": prediction == llm_sentiment,
        "topk_recall": round(topk_recall, 6),
        "jaccard": round(jaccard, 6),
        "signed_cosine": round(signed_cosine, 6),
        "evidence_mass": round(evidence_mass, 6),
        "polarity_agreement": round(polarity_agreement, 6),
        "llm_match_score": round(match, 6),
        "xai_top_words": " | ".join(words[idx] for idx in sorted(xai_top)),
        "llm_evidence_words": " | ".join(words[idx] for idx in sorted(llm_indices)),
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
        "llm_match_score",
    ]
    for (model, method), group_rows in grouped.items():
        item: dict[str, Any] = {"model": model, "method": method, "n_samples": len(group_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows]
            item[f"mean_{metric}"] = round(statistics.mean(values), 6)
        item["prediction_llm_agreement_rate"] = round(
            statistics.mean(1.0 if row["prediction_llm_agreement"] else 0.0 for row in group_rows),
            6,
        )
        summary.append(item)

    summary.sort(key=lambda row: float(row["mean_llm_match_score"]), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def mean_float(rows: list[dict[str, Any]], field: str) -> float:
    """Return a mean for numeric dict fields, or 0.0 for empty inputs."""
    if not rows:
        return 0.0
    return statistics.mean(float(row[field]) for row in rows)


def format_float(value: Any, digits: int = 3) -> str:
    """Format numeric values compactly for markdown tables."""
    return f"{float(value):.{digits}f}"


def group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group score rows by one or more string fields."""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return grouped


def write_evaluation_report(
    path: Path,
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    llm_by_id: dict[str, dict[str, Any]],
    top_k: int,
    top_n: int,
) -> None:
    """Write the main markdown evaluation report for the LLM-XAI experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted({str(row["sample_id"]) for row in score_rows})
    model_groups = group_rows(score_rows, "model")
    best_rows = sorted(score_rows, key=lambda row: float(row["llm_match_score"]), reverse=True)
    worst_rows = sorted(score_rows, key=lambda row: float(row["llm_match_score"]))

    lines: list[str] = [
        "# LLM-XAI Evaluation Report",
        "",
        "## Scope",
        "",
        f"- LLM evidence samples: {len(sample_ids)}",
        f"- Compared XAI rows: {len(score_rows)}",
        f"- Compared model-method pairs: {len(summary_rows)}",
        f"- XAI top-k used for overlap: {top_k}",
        "",
        "This report treats the LLM explanation as a qualitative reference, not as ground truth.",
        "The main question is whether each XAI method highlights the same eojeol-level evidence and polarity direction as the LLM response.",
        "",
        "## Metric Guide",
        "",
        "- `topk_recall`: share of LLM evidence words included in XAI top-k words.",
        "- `jaccard`: set overlap between LLM evidence words and XAI top-k words.",
        "- `signed_cosine`: polarity-aware cosine between the LLM evidence vector and the XAI score vector.",
        "- `evidence_mass`: total absolute XAI score mass assigned to LLM evidence words.",
        "- `polarity_agreement`: sign agreement on overlapping evidence words.",
        "- `llm_match_score`: presentation score combining the above metrics; use it for ranking, not as a truth label.",
        "",
        "## Overall Ranking",
        "",
        "| rank | model | method | n | match | top-k recall | jaccard | signed cosine | evidence mass | polarity agreement | pred agreement |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            f"{row['rank']} | {row['model']} | {row['method']} | {row['n_samples']} | "
            f"{format_float(row['mean_llm_match_score'])} | "
            f"{format_float(row['mean_topk_recall'])} | "
            f"{format_float(row['mean_jaccard'])} | "
            f"{format_float(row['mean_signed_cosine'])} | "
            f"{format_float(row['mean_evidence_mass'])} | "
            f"{format_float(row['mean_polarity_agreement'])} | "
            f"{format_float(row['prediction_llm_agreement_rate'])} |"
        )

    lines.extend(["", "## Model-Level Summary", ""])
    lines.extend(
        [
            "| model | n rows | mean match | mean signed cosine | mean evidence mass | prediction agreement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for (model,), rows in sorted(model_groups.items()):
        lines.append(
            "| "
            f"{model} | {len(rows)} | "
            f"{format_float(mean_float(rows, 'llm_match_score'))} | "
            f"{format_float(mean_float(rows, 'signed_cosine'))} | "
            f"{format_float(mean_float(rows, 'evidence_mass'))} | "
            f"{format_float(statistics.mean(1.0 if row['prediction_llm_agreement'] else 0.0 for row in rows))} |"
        )

    if summary_rows:
        best_method = summary_rows[0]
        weakest_method = summary_rows[-1]
        lines.extend(
            [
                "",
                "## Key Findings",
                "",
                f"- Best aligned method by `mean_llm_match_score`: `{best_method['model']} / {best_method['method']}` "
                f"({format_float(best_method['mean_llm_match_score'])}).",
                f"- Weakest aligned method by `mean_llm_match_score`: `{weakest_method['model']} / {weakest_method['method']}` "
                f"({format_float(weakest_method['mean_llm_match_score'])}).",
            ]
        )
        if len(model_groups) > 1:
            model_means = sorted(
                (
                    (model[0], mean_float(rows, "llm_match_score"))
                    for model, rows in model_groups.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            lines.append(
                f"- Strongest model family by average match: `{model_means[0][0]}` "
                f"({format_float(model_means[0][1])})."
            )

    lines.extend(
        [
            "",
            "## Best Aligned Rows",
            "",
            "| sample | model | method | match | LLM evidence | XAI top words |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in best_rows[:top_n]:
        lines.append(
            "| "
            f"{row['sample_id']} | {row['model']} | {row['method']} | "
            f"{format_float(row['llm_match_score'])} | {row['llm_evidence_words']} | {row['xai_top_words']} |"
        )

    lines.extend(
        [
            "",
            "## Weak Or Divergent Rows",
            "",
            "| sample | model | method | match | signed cosine | LLM evidence | XAI top words |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in worst_rows[:top_n]:
        lines.append(
            "| "
            f"{row['sample_id']} | {row['model']} | {row['method']} | "
            f"{format_float(row['llm_match_score'])} | {format_float(row['signed_cosine'])} | "
            f"{row['llm_evidence_words']} | {row['xai_top_words']} |"
        )

    lines.extend(["", "## Per-Sample Assessment", ""])
    by_sample = group_rows(score_rows, "sample_id")
    for sample_id in sample_ids:
        sample_rows = sorted(
            by_sample[(sample_id,)],
            key=lambda row: float(row["llm_match_score"]),
            reverse=True,
        )
        llm = llm_by_id[sample_id]
        evidence = ", ".join(str(item["phrase"]) for item in llm.get("evidence", []))
        agreement_count = sum(1 for row in sample_rows if row["prediction_llm_agreement"])
        lines.extend(
            [
                f"### {sample_id}",
                "",
                f"- text: {llm.get('text', '')}",
                f"- LLM sentiment: {llm['llm_sentiment']}",
                f"- LLM evidence: {evidence}",
                f"- model prediction agreement: {agreement_count}/{len(sample_rows)} rows",
                f"- best XAI: `{sample_rows[0]['model']} / {sample_rows[0]['method']}` "
                f"({format_float(sample_rows[0]['llm_match_score'])})",
                f"- weakest XAI: `{sample_rows[-1]['model']} / {sample_rows[-1]['method']}` "
                f"({format_float(sample_rows[-1]['llm_match_score'])})",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_report(
    path: Path,
    score_rows: list[dict[str, Any]],
    llm_by_id: dict[str, dict[str, Any]],
) -> None:
    """Write a compact qualitative markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_sample[str(row["sample_id"])].append(row)

    lines = ["# LLM-XAI Qualitative Case Report", ""]
    for sample_id in sorted(by_sample):
        llm = llm_by_id[sample_id]
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"text: {llm.get('text', '')}",
                "",
                f"- LLM sentiment: {llm['llm_sentiment']}",
                f"- LLM evidence: {', '.join(str(e['phrase']) for e in llm.get('evidence', []))}",
                f"- LLM reason: {llm.get('brief_reason', '')}",
                "",
                "| model | method | prediction | top words | score |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for row in sorted(by_sample[sample_id], key=lambda item: float(item["llm_match_score"]), reverse=True):
            lines.append(
                f"| {row['model']} | {row['method']} | {row['prediction']} | "
                f"{row['xai_top_words']} | {row['llm_match_score']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare LLM vectors with unified XAI outputs.")
    parser.add_argument("--xai", type=Path, default=comparison_dir() / "xai_unified.jsonl")
    parser.add_argument("--llm", type=Path, default=comparison_dir() / "llm_vectors.jsonl")
    parser.add_argument("--scores-output", type=Path, default=comparison_dir() / "llm_xai_overlap_scores.csv")
    parser.add_argument("--summary-output", type=Path, default=comparison_dir() / "llm_xai_method_summary.csv")
    parser.add_argument("--report-output", type=Path, default=comparison_dir() / "qualitative_case_report.md")
    parser.add_argument(
        "--evaluation-report-output",
        type=Path,
        default=comparison_dir() / "llm_xai_evaluation_report.md",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--report-top-n", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    xai_rows = read_jsonl(args.xai)
    llm_rows = read_jsonl(args.llm)
    llm_by_id = {str(row["sample_id"]): row for row in llm_rows}
    score_rows = [
        compare_one(row, llm_by_id[str(row["sample_id"])], args.top_k)
        for row in xai_rows
        if str(row["sample_id"]) in llm_by_id
    ]
    summary_rows = summarize(score_rows)
    write_csv(args.scores_output, score_rows, SCORE_FIELDS)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    write_case_report(args.report_output, score_rows, llm_by_id)
    write_evaluation_report(
        args.evaluation_report_output,
        score_rows,
        summary_rows,
        llm_by_id,
        args.top_k,
        args.report_top_n,
    )
    print(f"Wrote {args.scores_output} ({len(score_rows)} rows)")
    print(f"Wrote {args.summary_output} ({len(summary_rows)} rows)")
    print(f"Wrote {args.report_output}")
    print(f"Wrote {args.evaluation_report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
