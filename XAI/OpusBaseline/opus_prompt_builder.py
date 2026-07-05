"""Build Opus prompts for sentiment-evidence collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.OpusBaseline.common import (  # noqa: E402
    build_selected_reviews_from_texts,
    comparison_dir,
    read_selected_reviews,
    read_text_lines,
    write_jsonl,
    write_selected_reviews,
)


DEFAULT_PROMPT_VERSION = "opus_sentiment_evidence_v1"


def build_prompt(payload: dict[str, Any]) -> str:
    """Create one Korean prompt that asks for strict JSON evidence output."""
    input_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""다음 한국어 영화 리뷰가 positive인지 negative인지 판단하세요.

중요:
- 출력은 JSON object 하나만 작성하세요.
- sentiment는 "positive" 또는 "negative" 중 하나만 사용하세요.
- evidence는 최대 3개까지 작성하세요.
- evidence는 반드시 제공된 words 배열의 어절 index로 지정하세요.
- phrase는 원문에 실제로 등장하는 표현을 사용하세요.
- polarity는 "positive" 또는 "negative" 중 하나만 사용하세요.
- strength는 0.0~1.0 사이 숫자로 작성하세요.
- 이 설명은 정답 라벨이 아니라 XAI 비교용 사후 자연어 기준으로 사용됩니다.

입력:
{input_json}

출력 schema:
{{
  "sample_id": "{payload['sample_id']}",
  "text": "{payload['text']}",
  "words": {json.dumps(payload['words'], ensure_ascii=False)},
  "sentiment": "positive|negative",
  "evidence": [
    {{
      "phrase": "...",
      "word_indices": [0, 1],
      "polarity": "positive|negative",
      "strength": 0.8,
      "reason": "..."
    }}
  ],
  "brief_reason": "..."
}}"""


def load_or_create_selected_reviews(selected_reviews: Path, input_file: Path) -> list[dict[str, str]]:
    """Read selected reviews, or create them from a plain text input file."""
    if selected_reviews.exists():
        return read_selected_reviews(selected_reviews)

    texts = read_text_lines(input_file)
    if not texts:
        raise ValueError(f"No input texts found in {input_file}")
    rows = build_selected_reviews_from_texts(texts)
    write_selected_reviews(selected_reviews, rows)
    return rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build Opus prompt JSONL from selected reviews.")
    parser.add_argument(
        "--selected-reviews",
        type=Path,
        default=comparison_dir() / "selected_reviews.csv",
    )
    parser.add_argument("--input-file", type=Path, default=Path("inputs.txt"))
    parser.add_argument("--output", type=Path, default=comparison_dir() / "opus_prompts.jsonl")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build prompt records."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = load_or_create_selected_reviews(args.selected_reviews, args.input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    prompt_rows: list[dict[str, Any]] = []
    for row in rows:
        words = str(row["text"]).split()
        payload = {
            "sample_id": row["sample_id"],
            "text": row["text"],
            "words": words,
        }
        prompt_rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt_version": args.prompt_version,
                "text": row["text"],
                "words": words,
                "prompt": build_prompt(payload),
            }
        )

    write_jsonl(args.output, prompt_rows)
    print(f"Wrote {args.selected_reviews} ({len(rows)} rows)")
    print(f"Wrote {args.output} ({len(prompt_rows)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
