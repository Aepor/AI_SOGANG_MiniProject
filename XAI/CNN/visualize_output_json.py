"""Visualize CNN XAI JSON outputs.

The exporter writes one JSON file per method under ``XAI/CNN/outputs``. Each
JSON item contains whitespace-level ``words`` and one common ``scores`` array.
This script turns those arrays into presentation-friendly PNG charts and a
small HTML index. Legacy ``*_scores`` keys are still accepted for old exports.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


POSITIVE_COLOR = "#d95f5f"
NEGATIVE_COLOR = "#4c78a8"
ZERO_COLOR = "#9aa0a6"
GRID_COLOR = "#d9dee7"

METHOD_LABELS = {
    "filter_activation_scores": "Filter Activation",
    "integrated_gradients_scores": "Integrated Gradients",
    "ngram_occlusion_scores": "N-gram Occlusion",
    "unigram_occlusion_scores": "Unigram Occlusion",
    "scores": "Scores",
}


@dataclass(frozen=True)
class OutputFile:
    path: Path
    label: str
    score_key: str
    items: list[dict[str, Any]]


def configure_fonts() -> None:
    """Use a Korean-capable font when one is available."""
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "AppleGothic",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def find_score_key(item: dict[str, Any]) -> str:
    """Return the score array key in one JSON item."""
    if "scores" in item:
        return "scores"
    score_keys = [key for key in item if key.endswith("_scores")]
    if len(score_keys) != 1:
        raise ValueError(f"Expected scores or one *_scores key, found {score_keys}")
    return score_keys[0]


def label_from_filename(path: Path) -> str:
    """Infer a method label from an output filename."""
    stem = path.stem.replace("output_cnn_", "")
    if stem == "filter_activation":
        return "Filter Activation"
    if stem == "ngram_occlusion":
        return "N-gram Occlusion"
    if stem == "unigram_occlusion":
        return "Unigram Occlusion"
    if stem == "integrated_gradients":
        return "Integrated Gradients"
    if stem.startswith("integrated_gradients_steps"):
        steps = stem.replace("integrated_gradients_steps", "")
        return f"Integrated Gradients {steps}"
    return stem.replace("_", " ").title()


def method_label(path: Path, score_key: str, first_item: dict[str, Any]) -> str:
    """Build a readable chart label from the score key and filename."""
    if score_key == "scores":
        return label_from_filename(path)
    base_label = METHOD_LABELS.get(score_key, score_key.replace("_", " "))
    if score_key == "integrated_gradients_scores":
        steps = first_item.get("ig_steps")
        if steps:
            suffix = " default" if path.stem == "output_cnn_integrated_gradients" else ""
            return f"{base_label} {steps}{suffix}"
    return base_label


def load_output_file(path: Path) -> OutputFile:
    """Load one output JSON file and infer its score schema."""
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} does not contain a non-empty JSON list.")
    first_item = items[0]
    score_key = find_score_key(first_item)
    return OutputFile(
        path=path,
        label=method_label(path, score_key, first_item),
        score_key=score_key,
        items=items,
    )


def load_outputs(input_dir: Path, include_compatibility_files: bool = False) -> list[OutputFile]:
    """Load every output_cnn_*.json file in a deterministic order."""
    paths = sorted(input_dir.glob("output_cnn_*.json"))
    if not include_compatibility_files and any(
        path.name.startswith("output_cnn_integrated_gradients_steps") for path in paths
    ):
        paths = [path for path in paths if path.name != "output_cnn_integrated_gradients.json"]
    if not paths:
        raise FileNotFoundError(f"No output_cnn_*.json files found in {input_dir}")
    return [load_output_file(path) for path in paths]


def as_scores(item: dict[str, Any], score_key: str) -> tuple[list[str], list[float]]:
    """Return aligned words and scores for one JSON item."""
    words = [str(word) for word in item.get("words", [])]
    scores = [float(score) for score in item.get(score_key, [])]
    width = min(len(words), len(scores))
    return words[:width], scores[:width]


def short_text(value: Any, max_chars: int = 72) -> str:
    """Keep titles readable without hiding the sample identity."""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def wrap_label(value: str, width: int = 16) -> str:
    """Wrap long eojeol labels for chart axes."""
    if len(value) <= width:
        return value
    chunks = [value[index : index + width] for index in range(0, len(value), width)]
    return "\n".join(chunks[:3])


def safe_filename(value: str) -> str:
    """Make a stable filesystem-safe filename stem."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "chart"


def symmetric_limit(scores: list[float]) -> float:
    """Choose a symmetric x-axis limit around zero."""
    max_abs = max((abs(score) for score in scores), default=0.0)
    if max_abs == 0.0:
        return 1.0
    return max_abs * 1.18


def sample_label(item: dict[str, Any], item_index: int) -> str:
    """Return a stable label for charts without writing it into JSON."""
    return str(item.get("sample_id") or f"case_{item_index:03d}")


def draw_bar_chart(
    output: OutputFile, item: dict[str, Any], item_index: int, chart_dir: Path
) -> Path:
    """Draw one method/sample horizontal bar chart."""
    words, scores = as_scores(item, output.score_key)
    if not words:
        raise ValueError(f"{output.path.name} item {item_index} has no plottable scores.")

    colors = [POSITIVE_COLOR if score > 0 else NEGATIVE_COLOR if score < 0 else ZERO_COLOR for score in scores]
    sample_id = sample_label(item, item_index)
    filename = safe_filename(f"{output.path.stem}__{sample_id}.png")
    out_path = chart_dir / filename

    height = max(3.2, 0.48 * len(words) + 1.8)
    fig, ax = plt.subplots(figsize=(10.8, height))
    y_positions = list(range(len(words)))
    ax.barh(y_positions, scores, color=colors, height=0.64)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([wrap_label(word) for word in words])
    ax.invert_yaxis()
    ax.axvline(0.0, color="#20242a", linewidth=0.9)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-symmetric_limit(scores), symmetric_limit(scores))
    ax.set_xlabel("Signed normalized contribution")

    prediction = item.get("prediction", "")
    probability = item.get("probability", "")
    title_bits = [output.label, sample_id]
    if prediction != "":
        title_bits.append(f"pred={prediction}")
    if probability != "":
        title_bits.append(f"p={probability}")
    ax.set_title(" | ".join(title_bits), fontsize=12, pad=12)
    fig.text(0.5, 0.015, short_text(item.get("text", "")), ha="center", fontsize=9)

    x_limit = symmetric_limit(scores)
    label_offset = x_limit * 0.025
    for y_pos, score in zip(y_positions, scores):
        ha = "left" if score >= 0 else "right"
        x_pos = score + label_offset if score >= 0 else score - label_offset
        ax.text(x_pos, y_pos, f"{score:.3f}", va="center", ha=ha, fontsize=8)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.27, right=0.96, top=0.86, bottom=0.17)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def items_by_sample(output: OutputFile) -> dict[str, dict[str, Any]]:
    """Index JSON items by sample_id."""
    return {sample_label(item, index): item for index, item in enumerate(output.items, start=1)}


def draw_comparison_chart(
    sample_id: str,
    outputs: list[OutputFile],
    comparison_dir: Path,
) -> Path | None:
    """Draw one sample-level method comparison heatmap."""
    indexed = [(output, items_by_sample(output).get(sample_id)) for output in outputs]
    indexed = [(output, item) for output, item in indexed if item is not None]
    if len(indexed) < 2:
        return None

    words, _scores = as_scores(indexed[0][1], indexed[0][0].score_key)
    if not words:
        return None

    matrix: list[list[float]] = []
    row_labels: list[str] = []
    for output, item in indexed:
        _words, scores = as_scores(item, output.score_key)
        padded = scores[: len(words)] + [0.0] * max(0, len(words) - len(scores))
        matrix.append(padded[: len(words)])
        row_labels.append(output.label)

    max_abs = max((abs(score) for row in matrix for score in row), default=0.0)
    if max_abs == 0.0:
        max_abs = 1.0

    width = max(8.0, 0.72 * len(words) + 4.2)
    height = max(4.2, 0.48 * len(matrix) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(list(range(len(words))))
    ax.set_xticklabels([wrap_label(word, 10) for word in words], rotation=35, ha="right")
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels)
    ax.set_title(f"Method comparison | {sample_id}", fontsize=12, pad=12)

    for row_idx, row in enumerate(matrix):
        for col_idx, score in enumerate(row):
            if math.isfinite(score):
                color = "white" if abs(score) > max_abs * 0.55 else "#20242a"
                ax.text(col_idx, row_idx, f"{score:.3f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.subplots_adjust(left=0.26, right=0.95, top=0.86, bottom=0.28)

    out_path = comparison_dir / f"comparison__{safe_filename(sample_id)}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def relative_to(path: Path, base: Path) -> str:
    """Return a POSIX-style relative path for HTML links."""
    return path.relative_to(base).as_posix()


def write_html_index(
    out_path: Path,
    input_dir: Path,
    chart_paths: dict[str, list[Path]],
    comparison_paths: list[Path],
) -> None:
    """Write a browsable HTML index for the generated images."""
    sections: list[str] = [
        "<!doctype html>",
        "<html lang=\"ko\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>CNN XAI Output Visualizations</title>",
        "<style>",
        "body{font-family:Arial,'Malgun Gothic',sans-serif;margin:24px;background:#f6f8fb;color:#1f2933}",
        "h1,h2{margin:0 0 14px}",
        "section{margin:26px 0}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
        "figure{margin:0;background:#fff;border:1px solid #dce3ee;border-radius:8px;padding:10px}",
        "img{max-width:100%;height:auto;display:block}",
        "figcaption{font-size:12px;color:#52606d;margin-top:8px;word-break:break-all}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>CNN XAI Output Visualizations</h1>",
        f"<p>Source JSON directory: <code>{html.escape(str(input_dir))}</code></p>",
    ]

    if comparison_paths:
        sections.append("<section><h2>Method Comparisons</h2><div class=\"grid\">")
        for path in comparison_paths:
            link = html.escape(relative_to(path, out_path.parent))
            sections.append(
                f"<figure><img src=\"{link}\" alt=\"{html.escape(path.stem)}\">"
                f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
            )
        sections.append("</div></section>")

    for source_name, paths in chart_paths.items():
        sections.append(f"<section><h2>{html.escape(source_name)}</h2><div class=\"grid\">")
        for path in paths:
            link = html.escape(relative_to(path, out_path.parent))
            sections.append(
                f"<figure><img src=\"{link}\" alt=\"{html.escape(path.stem)}\">"
                f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
            )
        sections.append("</div></section>")

    sections.extend(["</body>", "</html>"])
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def write_summary(
    out_path: Path,
    input_dir: Path,
    output_dir: Path,
    outputs: list[OutputFile],
    chart_paths: dict[str, list[Path]],
    comparison_paths: list[Path],
    html_path: Path,
) -> None:
    """Write a machine-readable run summary."""
    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "json_files": [output.path.name for output in outputs],
        "sample_chart_count": sum(len(paths) for paths in chart_paths.values()),
        "comparison_chart_count": len(comparison_paths),
        "index_html": str(html_path),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize CNN XAI output_cnn_*.json files.")
    parser.add_argument("--input-dir", type=Path, default=Path("XAI/CNN/outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-compatibility-files",
        action="store_true",
        help="Also visualize compatibility files such as output_cnn_integrated_gradients.json.",
    )
    return parser.parse_args()


def main() -> int:
    """Run visualization export."""
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "visualizations"
    chart_dir = output_dir / "charts"
    comparison_dir = output_dir / "comparisons"
    chart_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    outputs = load_outputs(input_dir, args.include_compatibility_files)

    chart_paths: dict[str, list[Path]] = {}
    for output in outputs:
        paths: list[Path] = []
        for item_index, item in enumerate(output.items, start=1):
            paths.append(draw_bar_chart(output, item, item_index, chart_dir))
        chart_paths[output.path.name] = paths

    sample_ids = sorted(
        {
            sample_label(item, item_index)
            for output in outputs
            for item_index, item in enumerate(output.items, start=1)
        }
    )
    comparison_paths = [
        path
        for sample_id in sample_ids
        if (path := draw_comparison_chart(sample_id, outputs, comparison_dir)) is not None
    ]

    html_path = output_dir / "index.html"
    summary_path = output_dir / "visualization_summary.json"
    write_html_index(html_path, input_dir, chart_paths, comparison_paths)
    write_summary(summary_path, input_dir, output_dir, outputs, chart_paths, comparison_paths, html_path)

    print(f"Loaded {len(outputs)} JSON files from {input_dir}")
    print(f"Wrote {sum(len(paths) for paths in chart_paths.values())} sample charts")
    print(f"Wrote {len(comparison_paths)} comparison charts")
    print(f"Wrote {html_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
