"""CLI entrypoint that runs all CNN XAI methods.

이 파일은 방법별 모듈을 연결하는 orchestration layer이다. 실제 XAI 계산 로직은
`unigram_occlusion.py`, `ngram_occlusion.py`, `filter_activation.py`,
`maxpool_positions.py`, `gradient_x_input.py`, `integrated_gradients.py`에
나누어져 있고, 여기서는 다음 일만 한다.

1. argument parsing
2. vocab/model loading
3. 분석 sample 선정
4. 각 XAI 방법 실행
5. CSV/markdown/figure 저장
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from XAI.CNN.xai_methods.filter_activation import (  # noqa: E402
    compute_filter_class_direction,
    run_filter_activation_analysis,
)
from XAI.CNN.xai_methods.gradient_x_input import run_gradient_x_input  # noqa: E402
from XAI.CNN.xai_methods.integrated_gradients import run_integrated_gradients  # noqa: E402
from XAI.CNN.xai_methods.loader import build_or_load_vocab_cache, load_cnn_model  # noqa: E402
from XAI.CNN.xai_methods.maxpool_positions import run_maxpool_position_analysis  # noqa: E402
from XAI.CNN.xai_methods.model import predict_batch_ids, predict_one  # noqa: E402
from XAI.CNN.xai_methods.ngram_occlusion import run_ngram_occlusion  # noqa: E402
from XAI.CNN.xai_methods.unigram_occlusion import run_unigram_occlusion  # noqa: E402
from XAI.CNN.shared.nsmc_data import clean_nsmc_frame, load_nsmc_raw, sample_test_pool  # noqa: E402
from XAI.CNN.shared.paths import (  # noqa: E402
    default_cnn_cache_path,
    default_cnn_model_path,
    default_cnn_output_dir,
)
from XAI.CNN.shared.schemas import (  # noqa: E402
    LABEL_NAMES_KO,
    SampleRecord,
    format_float,
    make_sample_record,
    markdown_table,
    sample_record_to_row,
    write_csv,
)
from XAI.CNN.shared.tokenization import encode_tokens, make_okt, tokenize_text  # noqa: E402
from XAI.CNN.shared.visualization import HAS_MATPLOTLIB, plot_bar, safe_filename  # noqa: E402


DEFAULT_CUSTOM_TEXTS = [
    "이 영화 진짜 시간 가는 줄 모르고 봤어요",
    "완전 최악이고 시간이 아까웠다",
    "배우는 좋았지만 스토리는 너무 지루했다",
    "생각보다 지루하지 않고 감동적이었다",
    "재미는 있는데 결말이 별로였다",
]


def log(message: str) -> None:
    """Print progress immediately so long runs do not look frozen."""
    print(message, flush=True)


def choose_device(requested: str) -> torch.device:
    """Resolve the requested device string into a PyTorch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def select_samples(
    model: Any,
    word_to_index: dict[str, int],
    max_len: int,
    device: torch.device,
    sample_size: int,
    pool_size: int,
    case_per_group: int,
    seed: int,
    batch_size: int,
    custom_texts: list[str],
    only_custom: bool,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Select samples for batch XAI and smaller case-report examples.

    Returns:
        selected: 실제 XAI CSV 계산에 사용하는 sample 목록.
        case_records: markdown/figure에 자세히 보여 줄 대표 sample 목록.

    `selected`에는 random test samples + 대표 TP/TN/FP/FN + custom 문장이 합쳐진다.
    `case_records`는 보고서에서 읽기 좋은 예시만 담는다.
    """
    okt = make_okt()
    selected: list[SampleRecord] = []
    case_records: list[SampleRecord] = []

    pool_records: list[SampleRecord] = []
    if not only_custom:
        # test set에서 pool_size만큼 먼저 뽑고, 그 pool 안에서 random selected와
        # TP/TN/FP/FN 대표 사례를 고른다. 전체 test set을 XAI로 돌리면 너무 느리고 크다.
        _train_data, test_data, _train_path, _test_path = load_nsmc_raw()
        test_data = clean_nsmc_frame(test_data)
        pool = sample_test_pool(test_data, pool_size, seed)
        log(f"Tokenizing test candidate pool: {len(pool)} rows")
        token_rows: list[list[str]] = []
        id_rows: list[list[int]] = []
        lengths: list[int] = []
        for i, text in enumerate(pool["document"], start=1):
            tokens = tokenize_text(text, okt)
            ids, original_len = encode_tokens(tokens, word_to_index, max_len)
            token_rows.append(tokens[:max_len])
            id_rows.append(ids)
            lengths.append(original_len)
            if i % 500 == 0:
                log(f"  tokenized candidate rows: {i}/{len(pool)}")

        _logits, probs = predict_batch_ids(model, id_rows, device, batch_size)
        for (idx, row), tokens, ids, original_len, prob in zip(
            pool.iterrows(), token_rows, id_rows, lengths, probs
        ):
            pred = int(torch.argmax(prob).item())
            pred_info = {
                "pred": pred,
                "neg_prob": float(prob[0].item()),
                "pos_prob": float(prob[1].item()),
                "confidence": float(prob[pred].item()),
            }
            pool_records.append(
                make_sample_record(
                    sample_id=f"test_{idx}",
                    source="test",
                    original_index=str(idx),
                    text=str(row["document"]),
                    true_label=int(row["label"]),
                    tokens=tokens[:max_len],
                    ids=ids,
                    original_len=original_len,
                    pred_info=pred_info,
                )
            )

        random.Random(seed).shuffle(pool_records)
        selected.extend(pool_records[: min(sample_size, len(pool_records))])

        # 모델이 맞춘 사례와 틀린 사례를 같이 보고서에 넣기 위해 outcome별로 뽑는다.
        grouped: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in pool_records:
            grouped[record.outcome].append(record)
        for group in ["TP", "TN", "FP", "FN"]:
            group_records = sorted(
                grouped.get(group, []), key=lambda record: record.confidence, reverse=True
            )
            case_records.extend(group_records[:case_per_group])

    custom_records: list[SampleRecord] = []
    for i, text in enumerate(custom_texts, start=1):
        # custom 문장은 true label이 없으므로 outcome은 "custom"이다.
        tokens = tokenize_text(text, okt)
        ids, original_len = encode_tokens(tokens, word_to_index, max_len)
        pred_info = predict_one(model, ids, device, batch_size)
        custom_records.append(
            make_sample_record(
                sample_id=f"custom_{i}",
                source="custom",
                original_index="",
                text=text,
                true_label=None,
                tokens=tokens[:max_len],
                ids=ids,
                original_len=original_len,
                pred_info=pred_info,
            )
        )

    selected_by_id = {record.sample_id: record for record in selected}
    for record in case_records + custom_records:
        # case/custom sample도 반드시 CSV 계산 대상에 포함되도록 selected에 합친다.
        selected_by_id[record.sample_id] = record
    selected = list(selected_by_id.values())
    case_records = list({record.sample_id: record for record in case_records + custom_records}.values())
    log(f"Selected XAI samples: {len(selected)}; case samples: {len(case_records)}")
    return selected, case_records


def top_rows_for_sample(
    rows: list[dict[str, Any]],
    sample_id: str,
    score_key: str,
    top_k: int,
    positive_only: bool = False,
) -> list[dict[str, Any]]:
    """Get top rows for one sample by a chosen score column."""
    selected = [row for row in rows if row.get("sample_id") == sample_id]
    if positive_only:
        selected = [row for row in selected if float(row.get(score_key, 0.0)) > 0]
    return sorted(selected, key=lambda row: float(row.get(score_key, 0.0)), reverse=True)[:top_k]


def create_figures(
    figure_dir: Path,
    case_records: list[SampleRecord],
    unigram_rows: list[dict[str, Any]],
    ngram_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    integrated_rows: list[dict[str, Any]],
    maxpool_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    figure_limit: int,
) -> None:
    """Create the standard PNG figures used in reports.

    이 함수는 이미 계산된 CSV row들을 그림으로 바꾸기만 한다. 재예측/재분석은 하지 않는다.
    """
    if not HAS_MATPLOTLIB:
        log("matplotlib is not available; skipping figures.")
        return
    for record in case_records[:figure_limit]:
        prefix = safe_filename(record.sample_id)
        uni_top = top_rows_for_sample(unigram_rows, record.sample_id, "prob_drop", 12, True)
        plot_bar(
            [row["token"] for row in uni_top],
            [float(row["prob_drop"]) for row in uni_top],
            f"Unigram occlusion: {record.sample_id}",
            "target probability drop",
            figure_dir / f"unigram_occlusion_case_{prefix}.png",
            color="#4C78A8",
        )

        ngram_top = top_rows_for_sample(ngram_rows, record.sample_id, "prob_drop", 12, True)
        plot_bar(
            [f"{row['ngram']} (n={row['n']})" for row in ngram_top],
            [float(row["prob_drop"]) for row in ngram_top],
            f"n-gram occlusion: {record.sample_id}",
            "target probability drop",
            figure_dir / f"ngram_occlusion_case_{prefix}.png",
            color="#F58518",
        )

        grad_top = top_rows_for_sample(
            gradient_rows, record.sample_id, "normalized_abs_score", 12, False
        )
        plot_bar(
            [row["token"] for row in grad_top],
            [float(row["normalized_abs_score"]) for row in grad_top],
            f"Gradient x Input: {record.sample_id}",
            "normalized abs score",
            figure_dir / f"gradient_x_input_case_{prefix}.png",
            color="#54A24B",
        )

        ig_top = top_rows_for_sample(
            integrated_rows, record.sample_id, "normalized_abs_score", 12, False
        )
        plot_bar(
            [row["token"] for row in ig_top],
            [float(row["normalized_abs_score"]) for row in ig_top],
            f"Integrated Gradients: {record.sample_id}",
            "normalized abs score",
            figure_dir / f"integrated_gradients_case_{prefix}.png",
            color="#72B7B2",
        )

        maxpool_top = top_rows_for_sample(
            maxpool_rows, record.sample_id, "target_contribution", 12, True
        )
        plot_bar(
            [
                f"{row['selected_ngram']} (fs={row['filter_size']}, f={row['filter_idx']})"
                for row in maxpool_top
            ],
            [float(row["target_contribution"]) for row in maxpool_top],
            f"Max-pooling contribution: {record.sample_id}",
            "target contribution",
            figure_dir / f"maxpool_case_{prefix}.png",
            color="#B279A2",
        )

    top_filter_rows = sorted(
        filter_rows, key=lambda row: abs(float(row["target_contribution"])), reverse=True
    )[:20]
    plot_bar(
        [
            f"{row['ngram']} (fs={row['filter_size']}, f={row['filter_idx']})"
            for row in top_filter_rows
        ],
        [float(row["target_contribution"]) for row in top_filter_rows],
        "Top filter activation contributions",
        "target contribution",
        figure_dir / "filter_activation_top_patterns.png",
        color="#E45756",
    )


def create_case_summary(
    output_path: Path,
    case_records: list[SampleRecord],
    unigram_rows: list[dict[str, Any]],
    ngram_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    integrated_rows: list[dict[str, Any]],
    maxpool_rows: list[dict[str, Any]],
) -> None:
    """Write a compact markdown report for the selected case examples."""
    lines: list[str] = [
        "# CNN XAI Case Summary",
        "",
        "이 파일은 CNN XAI 모듈형 파이프라인이 생성한 사례 요약이다.",
        "",
    ]
    for record in case_records:
        lines.extend(
            [
                f"## {record.sample_id} ({record.outcome})",
                "",
                f"- text: {record.text}",
                f"- true label: {'' if record.true_label is None else LABEL_NAMES_KO[record.true_label]}",
                f"- predicted label: {LABEL_NAMES_KO[record.pred_label]}",
                f"- target class: {LABEL_NAMES_KO[record.target_class]}",
                f"- negative prob: {record.neg_prob:.4f}",
                f"- positive prob: {record.pos_prob:.4f}",
                f"- tokens: {' / '.join(record.tokens[:record.original_len])}",
                "",
            ]
        )

        uni_rows = [
            {"token": row["token"], "position": row["position"], "prob_drop": format_float(row["prob_drop"])}
            for row in top_rows_for_sample(unigram_rows, record.sample_id, "prob_drop", 5, True)
        ]
        lines.extend(["### Top unigram occlusion", markdown_table(uni_rows, ["token", "position", "prob_drop"]), ""])

        ngram_md_rows = [
            {
                "n": row["n"],
                "ngram": row["ngram"],
                "start_pos": row["start_pos"],
                "prob_drop": format_float(row["prob_drop"]),
            }
            for row in top_rows_for_sample(ngram_rows, record.sample_id, "prob_drop", 5, True)
        ]
        lines.extend(
            [
                "### Top n-gram occlusion",
                markdown_table(ngram_md_rows, ["n", "ngram", "start_pos", "prob_drop"]),
                "",
            ]
        )

        maxpool_md_rows = [
            {
                "filter_size": row["filter_size"],
                "filter_idx": row["filter_idx"],
                "selected_ngram": row["selected_ngram"],
                "target_contribution": format_float(row["target_contribution"]),
            }
            for row in top_rows_for_sample(
                maxpool_rows, record.sample_id, "target_contribution", 5, True
            )
        ]
        lines.extend(
            [
                "### Top max-pooling contributions",
                markdown_table(
                    maxpool_md_rows,
                    ["filter_size", "filter_idx", "selected_ngram", "target_contribution"],
                ),
                "",
            ]
        )

        grad_md_rows = [
            {
                "token": row["token"],
                "position": row["position"],
                "signed_score": format_float(row["signed_score"]),
                "normalized_abs_score": format_float(row["normalized_abs_score"]),
            }
            for row in top_rows_for_sample(
                gradient_rows, record.sample_id, "normalized_abs_score", 5, False
            )
        ]
        lines.extend(
            [
                "### Top Gradient x Input",
                markdown_table(
                    grad_md_rows,
                    ["token", "position", "signed_score", "normalized_abs_score"],
                ),
                "",
            ]
        )

        ig_md_rows = [
            {
                "token": row["token"],
                "position": row["position"],
                "signed_score": format_float(row["signed_score"]),
                "normalized_abs_score": format_float(row["normalized_abs_score"]),
            }
            for row in top_rows_for_sample(
                integrated_rows, record.sample_id, "normalized_abs_score", 5, False
            )
        ]
        lines.extend(
            [
                "### Top Integrated Gradients",
                markdown_table(
                    ig_md_rows,
                    ["token", "position", "signed_score", "normalized_abs_score"],
                ),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_ngram_sizes(value: str) -> list[int]:
    """Parse CLI value like `1,2,3,4,5` into integer sizes."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Define command-line options for the full CNN XAI pipeline."""
    parser = argparse.ArgumentParser(description="Run CNN-only XAI analysis for nsmc_cnn.ipynb.")
    parser.add_argument("--model-path", type=Path, default=default_cnn_model_path())
    parser.add_argument("--cache-path", type=Path, default=default_cnn_cache_path())
    parser.add_argument("--output-dir", type=Path, default=default_cnn_output_dir())
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--case-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ngram-sizes", default="1,2,3,4,5")
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=32,
        help="Number of integration intervals for CNN Integrated Gradients.",
    )
    parser.add_argument(
        "--skip-integrated-gradients",
        action="store_true",
        help="Skip Integrated Gradients if a quick run is needed.",
    )
    parser.add_argument("--custom-text", action="append", default=[])
    parser.add_argument("--no-custom", action="store_true")
    parser.add_argument("--only-custom", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--figure-limit", type=int, default=16)
    return parser.parse_args(argv)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV and log its row count."""
    write_csv(path, rows)
    log(f"Wrote {path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    """Run the full CNN XAI pipeline."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Seed를 고정하면 같은 pool/sample selection이 다시 나오므로 보고서 재현성이 좋아진다.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    log(f"Using device: {device}")

    cache = build_or_load_vocab_cache(args.cache_path, args.refresh_cache)
    word_to_index = cache["word_to_index"]
    vocab = cache["vocab"]
    log(f"Loaded vocab cache: {args.cache_path} (vocab={len(vocab)})")

    model, arch = load_cnn_model(args.model_path, len(vocab), device)
    log(
        "Loaded CNN model: "
        f"vocab={arch['vocab_size']}, emb={arch['embedding_dim']}, "
        f"filters={arch['filter_sizes']}x{arch['n_filters']}"
    )
    if args.max_len < max(arch["filter_sizes"]):
        raise ValueError("--max-len must be at least the largest CNN filter size.")

    # 기본 custom 문장은 반전/명확 긍정/명확 부정/혼합 감정 예시를 빠르게 보기 위한 것이다.
    custom_texts = [] if args.no_custom else (args.custom_text or DEFAULT_CUSTOM_TEXTS)
    selected, case_records = select_samples(
        model=model,
        word_to_index=word_to_index,
        max_len=args.max_len,
        device=device,
        sample_size=args.sample_size,
        pool_size=max(args.pool_size, args.sample_size),
        case_per_group=args.case_per_group,
        seed=args.seed,
        batch_size=args.batch_size,
        custom_texts=custom_texts,
        only_custom=args.only_custom,
    )

    save_csv(output_dir / "cnn_xai_selected_samples.csv", [sample_record_to_row(r) for r in selected])
    pad_idx = word_to_index.get("<pad>", 0)
    ngram_sizes = parse_ngram_sizes(args.ngram_sizes)

    # 아래부터는 방법별 모듈을 순서대로 호출한다. 각 method는 같은 selected sample을 입력받고
    # method-specific CSV row 목록을 반환한다.
    log("Running unigram occlusion...")
    unigram_rows = run_unigram_occlusion(model, selected, pad_idx, device, args.batch_size)
    save_csv(output_dir / "cnn_unigram_occlusion.csv", unigram_rows)

    log("Running n-gram occlusion...")
    ngram_rows = run_ngram_occlusion(model, selected, pad_idx, ngram_sizes, device, args.batch_size)
    save_csv(output_dir / "cnn_ngram_occlusion.csv", ngram_rows)

    log("Running filter activation analysis...")
    filter_rows = run_filter_activation_analysis(model, selected, device)
    save_csv(output_dir / "cnn_filter_top_ngrams.csv", filter_rows)
    direction_rows = compute_filter_class_direction(model)
    save_csv(output_dir / "cnn_filter_class_direction.csv", direction_rows)

    log("Running max-pooling position analysis...")
    maxpool_rows = run_maxpool_position_analysis(model, selected, device)
    save_csv(output_dir / "cnn_maxpool_positions.csv", maxpool_rows)

    log("Running Gradient x Input...")
    gradient_rows = run_gradient_x_input(model, selected, device)
    save_csv(output_dir / "cnn_gradient_x_input.csv", gradient_rows)

    if args.skip_integrated_gradients:
        log("Skipping Integrated Gradients.")
        integrated_rows: list[dict[str, Any]] = []
    else:
        log(f"Running Integrated Gradients... (steps={args.ig_steps})")
        integrated_rows = run_integrated_gradients(
            model=model,
            samples=selected,
            pad_idx=pad_idx,
            device=device,
            steps=args.ig_steps,
        )
        save_csv(output_dir / "cnn_integrated_gradients.csv", integrated_rows)

    summary_path = output_dir / "cnn_xai_case_summary.md"
    create_case_summary(
        summary_path,
        case_records,
        unigram_rows,
        ngram_rows,
        gradient_rows,
        integrated_rows,
        maxpool_rows,
    )
    log(f"Wrote {summary_path}")

    metadata = {
        "model_path": str(args.model_path.resolve()),
        "cache_path": str(args.cache_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "sample_size": args.sample_size,
        "pool_size": args.pool_size,
        "case_per_group": args.case_per_group,
        "max_len": args.max_len,
        "ngram_sizes": ngram_sizes,
        "integrated_gradients_enabled": not args.skip_integrated_gradients,
        "ig_steps": None if args.skip_integrated_gradients else args.ig_steps,
        "vocab_size": len(vocab),
        "architecture": arch,
        "selected_samples": len(selected),
        "case_samples": len(case_records),
        "pipeline": "modular_xai_methods",
    }
    metadata_path = output_dir / "cnn_xai_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote {metadata_path}")

    if not args.no_plots:
        create_figures(
            figure_dir=figure_dir,
            case_records=case_records,
            unigram_rows=unigram_rows,
            ngram_rows=ngram_rows,
            gradient_rows=gradient_rows,
            integrated_rows=integrated_rows,
            maxpool_rows=maxpool_rows,
            filter_rows=filter_rows,
            figure_limit=args.figure_limit,
        )

    log("CNN XAI analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
