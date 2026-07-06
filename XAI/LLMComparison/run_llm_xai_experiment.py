"""Run the full LLM-XAI comparison pipeline.

Pipeline:
1. unify CNN/FNN/Transformer XAI outputs
2. build LLM prompts
3. collect LLM explanations
4. normalize LLM evidence vectors
5. compare XAI with LLM evidence and write reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.LLMComparison import (  # noqa: E402
    collect_llm_explanations,
    compare_llm_with_xai,
    llm_prompt_builder,
    normalize_llm_outputs,
    unify_xai_outputs,
)
from XAI.LLMComparison.common import comparison_dir, xai_root  # noqa: E402


def run_stage(name: str, argv: list[str], dry_run: bool) -> None:
    """Run one pipeline stage using the module's main(argv) function."""
    print(f"[stage] {name}")
    print("  " + " ".join(argv))
    if dry_run:
        return

    stage_main = {
        "unify": unify_xai_outputs.main,
        "prompts": llm_prompt_builder.main,
        "collect": collect_llm_explanations.main,
        "normalize": normalize_llm_outputs.main,
        "compare": compare_llm_with_xai.main,
    }[name]
    exit_code = stage_main(argv)
    if exit_code != 0:
        raise RuntimeError(f"Stage failed: {name} ({exit_code})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    default_dir = comparison_dir()
    parser = argparse.ArgumentParser(description="Run the full LLM-XAI comparison experiment.")
    parser.add_argument("--models", default="cnn,fnn,transformer")
    parser.add_argument("--xai-output-dir", type=Path, default=xai_root() / "outputs_json")
    parser.add_argument("--selected-reviews", type=Path, default=default_dir / "selected_reviews.csv")
    parser.add_argument("--input-file", type=Path, default=Path("inputs.txt"))
    parser.add_argument("--xai-unified", type=Path, default=default_dir / "xai_unified.jsonl")
    parser.add_argument("--prompts", type=Path, default=default_dir / "llm_prompts.jsonl")
    parser.add_argument("--llm-explanations", type=Path, default=default_dir / "llm_explanations.jsonl")
    parser.add_argument("--llm-vectors", type=Path, default=default_dir / "llm_vectors.jsonl")
    parser.add_argument("--scores-output", type=Path, default=default_dir / "llm_xai_overlap_scores.csv")
    parser.add_argument("--summary-output", type=Path, default=default_dir / "llm_xai_method_summary.csv")
    parser.add_argument("--case-report-output", type=Path, default=default_dir / "qualitative_case_report.md")
    parser.add_argument(
        "--evaluation-report-output",
        type=Path,
        default=default_dir / "llm_xai_evaluation_report.md",
    )
    parser.add_argument("--collect-mode", choices=["api", "manual"], default="api")
    parser.add_argument("--manual-input", type=Path, default=default_dir / "llm_manual.jsonl")
    parser.add_argument("--model", default="")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--report-top-n", type=int, default=5)
    parser.add_argument("--overwrite-llm", action="store_true")
    parser.add_argument("--skip-unify", action="store_true")
    parser.add_argument("--skip-prompts", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the configured pipeline stages."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not args.skip_unify:
        run_stage(
            "unify",
            [
                "--models",
                args.models,
                "--output-dir",
                str(args.xai_output_dir),
                "--selected-reviews",
                str(args.selected_reviews),
                "--output",
                str(args.xai_unified),
            ],
            args.dry_run,
        )

    if not args.skip_prompts:
        run_stage(
            "prompts",
            [
                "--selected-reviews",
                str(args.selected_reviews),
                "--input-file",
                str(args.input_file),
                "--output",
                str(args.prompts),
            ],
            args.dry_run,
        )

    if not args.skip_collect:
        collect_argv = [
            "--mode",
            args.collect_mode,
            "--manual-input",
            str(args.manual_input),
            "--prompts",
            str(args.prompts),
            "--output",
            str(args.llm_explanations),
            "--max-samples",
            str(args.max_samples),
        ]
        if args.model:
            collect_argv.extend(["--model", args.model])
        if args.reasoning_effort:
            collect_argv.extend(["--reasoning-effort", args.reasoning_effort])
        if args.overwrite_llm:
            collect_argv.append("--overwrite")
        run_stage("collect", collect_argv, args.dry_run)

    if not args.skip_normalize:
        run_stage(
            "normalize",
            [
                "--input",
                str(args.llm_explanations),
                "--output",
                str(args.llm_vectors),
            ],
            args.dry_run,
        )

    if not args.skip_compare:
        run_stage(
            "compare",
            [
                "--xai",
                str(args.xai_unified),
                "--llm",
                str(args.llm_vectors),
                "--scores-output",
                str(args.scores_output),
                "--summary-output",
                str(args.summary_output),
                "--report-output",
                str(args.case_report_output),
                "--evaluation-report-output",
                str(args.evaluation_report_output),
                "--top-k",
                str(args.top_k),
                "--report-top-n",
                str(args.report_top_n),
            ],
            args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
