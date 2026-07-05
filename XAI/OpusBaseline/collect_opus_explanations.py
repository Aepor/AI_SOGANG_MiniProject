"""Collect Opus sentiment evidence outputs.

The first supported workflow is manual mode: paste Opus JSON objects into a
JSONL file and this script normalizes metadata into `opus_explanations.jsonl`.
API mode is intentionally left explicit so the experiment can run without
network credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.OpusBaseline.common import comparison_dir, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_MODEL_ID = "manual-opus"
DEFAULT_PROMPT_VERSION = "opus_sentiment_evidence_v1"


def parse_response_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Accept either a direct Opus JSON object or `{sample_id, response}`."""
    if "response" not in row:
        return dict(row)
    response = row["response"]
    if isinstance(response, dict):
        payload = dict(response)
    elif isinstance(response, str):
        payload = json.loads(response)
    else:
        raise ValueError("manual row response must be a JSON object or JSON string")
    if "sample_id" not in payload and "sample_id" in row:
        payload["sample_id"] = row["sample_id"]
    return payload


def collect_manual(manual_input: Path, prompts: Path, output: Path, overwrite: bool) -> int:
    """Collect manually pasted Opus outputs."""
    if not manual_input.exists():
        raise FileNotFoundError(
            f"Manual input not found: {manual_input}. "
            "Create a JSONL file with one Opus JSON object per line."
        )

    existing = [] if overwrite else read_jsonl(output)
    existing_by_id = {row.get("sample_id"): row for row in existing}
    prompts_by_id = {row.get("sample_id"): row for row in read_jsonl(prompts)}
    now = datetime.now(timezone.utc).isoformat()
    collected = list(existing_by_id.values())

    for row in read_jsonl(manual_input):
        payload = parse_response_payload(row)
        sample_id = payload.get("sample_id")
        if not sample_id:
            raise ValueError(f"Manual row is missing sample_id: {row}")
        if sample_id in existing_by_id and not overwrite:
            continue
        prompt_row = prompts_by_id.get(sample_id, {})
        payload.setdefault("text", prompt_row.get("text", ""))
        payload.setdefault("words", prompt_row.get("words", []))
        payload.setdefault("model_id", row.get("model_id", DEFAULT_MODEL_ID))
        payload.setdefault("prompt_version", row.get("prompt_version", DEFAULT_PROMPT_VERSION))
        payload.setdefault("created_at", row.get("created_at", now))
        existing_by_id[sample_id] = payload

    collected = list(existing_by_id.values())
    collected.sort(key=lambda item: str(item.get("sample_id", "")))
    write_jsonl(output, collected)
    print(f"Wrote {output} ({len(collected)} rows)")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Collect Opus explanation JSONL.")
    parser.add_argument("--mode", choices=["manual", "api"], default="manual")
    parser.add_argument(
        "--manual-input",
        type=Path,
        default=comparison_dir() / "opus_manual.jsonl",
    )
    parser.add_argument("--prompts", type=Path, default=comparison_dir() / "opus_prompts.jsonl")
    parser.add_argument("--output", type=Path, default=comparison_dir() / "opus_explanations.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run collection."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode == "api":
        raise RuntimeError(
            "API mode is not implemented yet. Use --mode manual with Opus JSONL output first."
        )
    return collect_manual(args.manual_input, args.prompts, args.output, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
