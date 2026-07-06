"""Collect LLM sentiment evidence outputs.

Supports two workflows:
- manual mode: paste JSON objects into a JSONL file
- api mode: call the OpenAI Responses API with prompts from llm_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.LLMComparison.common import (  # noqa: E402
    comparison_dir,
    load_dotenv,
    read_jsonl,
    write_jsonl,
)

load_dotenv()

DEFAULT_MANUAL_MODEL_ID = "manual-llm"
DEFAULT_OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL", "gpt-5.5-pro")
DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "high")
DEFAULT_PROMPT_VERSION = "llm_sentiment_evidence_v1"
DEFAULT_OPENAI_ENDPOINT = os.environ.get(
    "OPENAI_RESPONSES_URL",
    "https://api.openai.com/v1/responses",
)

SYSTEM_INSTRUCTIONS = (
    "You are a careful Korean sentiment-analysis annotator. "
    "Judge the review independently from any model XAI output. "
    "Return only the requested JSON object."
)

LLM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "text": {"type": "string"},
        "words": {"type": "array", "items": {"type": "string"}},
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string"},
                    "word_indices": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "integer"},
                    },
                    "polarity": {"type": "string", "enum": ["positive", "negative"]},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": ["phrase", "word_indices", "polarity", "strength", "reason"],
                "additionalProperties": False,
            },
        },
        "brief_reason": {"type": "string"},
    },
    "required": ["sample_id", "text", "words", "sentiment", "evidence", "brief_reason"],
    "additionalProperties": False,
}


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse a JSON object, accepting accidental fenced output in manual files."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def parse_response_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Accept either a direct LLM JSON object or `{sample_id, response}`."""
    if "response" in row:
        response = row["response"]
    elif "output_text" in row:
        response = row["output_text"]
    else:
        return dict(row)

    if isinstance(response, dict):
        payload = dict(response)
    elif isinstance(response, str):
        payload = parse_json_text(response)
    else:
        raise ValueError("manual row response must be a JSON object or JSON string")
    if "sample_id" not in payload and "sample_id" in row:
        payload["sample_id"] = row["sample_id"]
    return payload


def sorted_rows(rows_by_id: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows sorted by sample_id for stable JSONL output."""
    rows = list(rows_by_id.values())
    rows.sort(key=lambda item: str(item.get("sample_id", "")))
    return rows


def collect_manual(manual_input: Path, prompts: Path, output: Path, overwrite: bool) -> int:
    """Collect manually pasted LLM outputs."""
    if not manual_input.exists():
        raise FileNotFoundError(
            f"Manual input not found: {manual_input}. "
            "Create a JSONL file with one LLM JSON object per line."
        )

    existing = [] if overwrite else read_jsonl(output)
    existing_by_id = {row.get("sample_id"): row for row in existing}
    prompts_by_id = {row.get("sample_id"): row for row in read_jsonl(prompts)}
    now = datetime.now(timezone.utc).isoformat()

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
        payload.setdefault("model_id", row.get("model_id", DEFAULT_MANUAL_MODEL_ID))
        payload.setdefault("prompt_version", row.get("prompt_version", DEFAULT_PROMPT_VERSION))
        payload.setdefault("created_at", row.get("created_at", now))
        payload.setdefault("api_provider", row.get("api_provider", "manual"))
        existing_by_id[sample_id] = payload

    collected = sorted_rows(existing_by_id)
    write_jsonl(output, collected)
    print(f"Wrote {output} ({len(collected)} rows)")
    return 0


def build_openai_request(
    prompt_row: dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Build one OpenAI Responses API request body."""
    request_body = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": str(prompt_row["prompt"])},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sentiment_evidence",
                "strict": True,
                "schema": LLM_RESPONSE_SCHEMA,
            }
        },
    }
    if reasoning_effort:
        request_body["reasoning"] = {"effort": reasoning_effort}
    return request_body


def post_openai_response(
    endpoint: str,
    api_key: str,
    request_body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Call the OpenAI Responses API and return the decoded JSON body."""
    data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed ({exc.code}): {detail}") from exc
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("OpenAI API response must be a JSON object")
    return value


def extract_response_text(response: dict[str, Any]) -> str:
    """Extract text from common Responses API response shapes."""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "output_text"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    raise ValueError("Could not find output text in OpenAI API response")


def collect_api(
    prompts: Path,
    output: Path,
    overwrite: bool,
    model: str,
    endpoint: str,
    api_key: str,
    timeout: float,
    max_samples: int,
    reasoning_effort: str,
) -> int:
    """Collect LLM outputs by calling the OpenAI Responses API."""
    prompt_rows = read_jsonl(prompts)
    if not prompt_rows:
        raise ValueError(f"No prompt rows found in {prompts}")

    existing = [] if overwrite else read_jsonl(output)
    existing_by_id = {row.get("sample_id"): row for row in existing}
    new_calls = 0

    for prompt_row in prompt_rows:
        sample_id = prompt_row.get("sample_id")
        if not sample_id:
            raise ValueError(f"Prompt row missing sample_id: {prompt_row}")
        if sample_id in existing_by_id and not overwrite:
            continue
        if max_samples > 0 and new_calls >= max_samples:
            break

        response = post_openai_response(
            endpoint=endpoint,
            api_key=api_key,
            request_body=build_openai_request(prompt_row, model, reasoning_effort),
            timeout=timeout,
        )
        payload = parse_json_text(extract_response_text(response))
        payload.setdefault("sample_id", sample_id)
        payload.setdefault("text", prompt_row.get("text", ""))
        payload.setdefault("words", prompt_row.get("words", []))
        payload["model_id"] = response.get("model", model)
        payload["prompt_version"] = prompt_row.get("prompt_version", DEFAULT_PROMPT_VERSION)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        payload["api_provider"] = "openai"
        payload["response_id"] = response.get("id", "")
        existing_by_id[sample_id] = payload
        new_calls += 1

        # Save after each API call so a long run can resume without losing work.
        write_jsonl(output, sorted_rows(existing_by_id))

    collected = sorted_rows(existing_by_id)
    write_jsonl(output, collected)
    print(f"Wrote {output} ({len(collected)} rows, {new_calls} new API calls)")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Collect LLM explanation JSONL.")
    parser.add_argument("--mode", choices=["manual", "api"], default="api")
    parser.add_argument(
        "--manual-input",
        type=Path,
        default=comparison_dir() / "llm_manual.jsonl",
    )
    parser.add_argument("--prompts", type=Path, default=comparison_dir() / "llm_prompts.jsonl")
    parser.add_argument("--output", type=Path, default=comparison_dir() / "llm_explanations.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL_ID)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--endpoint", default=DEFAULT_OPENAI_ENDPOINT)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run collection."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode == "manual":
        return collect_manual(args.manual_input, args.prompts, args.output, args.overwrite)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    return collect_api(
        prompts=args.prompts,
        output=args.output,
        overwrite=args.overwrite,
        model=args.model,
        endpoint=args.endpoint,
        api_key=api_key,
        timeout=args.timeout,
        max_samples=args.max_samples,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    raise SystemExit(main())
