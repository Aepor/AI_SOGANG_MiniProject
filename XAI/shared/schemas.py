"""Shared row schemas and small formatting helpers.

XAI 결과는 여러 CSV와 markdown 파일로 흩어진다. 그래서 sample/prediction 관련
기본 필드를 `SampleRecord` 하나로 묶어 두면, 각 XAI 방법이 같은 입력 구조를 공유할 수
있다. 나중에 FNN/Transformer를 붙일 때도 이 schema를 기준으로 맞추면 비교가 쉬워진다.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# NSMC label convention: 0 = negative, 1 = positive.
LABEL_NAMES = {0: "negative", 1: "positive"}
LABEL_NAMES_KO = {0: "부정", 1: "긍정"}


@dataclass
class SampleRecord:
    """A fully prepared review sample used by every XAI method.

    Fields:
        sample_id: CSV/figure/report join에 사용하는 안정적인 id.
        source: `test` 또는 `custom`처럼 샘플 출처.
        original_index: test dataframe 원래 index. custom 문장은 빈 문자열.
        text: 원문 리뷰.
        true_label: NSMC 정답 label. custom 문장은 None.
        tokens: Okt + stopword 제거 후 token 목록.
        ids: padding까지 끝난 model input id sequence.
        original_len: ids 중 실제 token에 해당하는 길이. pad 위치 제외에 사용.
        pred_label: CNN 예측 class.
        neg_prob/pos_prob/confidence: softmax 확률.
        target_class: XAI가 설명할 class. 기본은 모델이 예측한 class.
        outcome: TP/TN/FP/FN/custom.
    """

    sample_id: str
    source: str
    original_index: str
    text: str
    true_label: int | None
    tokens: list[str]
    ids: list[int]
    original_len: int
    pred_label: int
    neg_prob: float
    pos_prob: float
    confidence: float
    target_class: int
    outcome: str


def outcome_name(true_label: int | None, pred_label: int) -> str:
    """Classify one prediction into report-friendly outcome buckets."""
    if true_label is None:
        return "custom"
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    if true_label == 1 and pred_label == 0:
        return "FN"
    return "unknown"


def make_sample_record(
    sample_id: str,
    source: str,
    original_index: str,
    text: str,
    true_label: int | None,
    tokens: list[str],
    ids: list[int],
    original_len: int,
    pred_info: dict[str, Any],
    target_class: int | None = None,
) -> SampleRecord:
    """Build a `SampleRecord` after model prediction.

    `target_class`를 따로 주지 않으면 "모델이 왜 이렇게 예측했는가"를 설명하기 위해
    predicted class를 target으로 둔다. 오분류 분석에서 true class 기준 설명을 보고 싶을
    때만 target_class를 명시하면 된다.
    """
    pred_label = int(pred_info["pred"])
    target = pred_label if target_class is None else int(target_class)
    return SampleRecord(
        sample_id=sample_id,
        source=source,
        original_index=original_index,
        text=text,
        true_label=true_label,
        tokens=tokens,
        ids=ids,
        original_len=original_len,
        pred_label=pred_label,
        neg_prob=float(pred_info["neg_prob"]),
        pos_prob=float(pred_info["pos_prob"]),
        confidence=float(pred_info["confidence"]),
        target_class=target,
        outcome=outcome_name(true_label, pred_label),
    )


def sample_record_to_row(record: SampleRecord) -> dict[str, Any]:
    """Flatten a `SampleRecord` into one CSV row."""
    return {
        "sample_id": record.sample_id,
        "source": record.source,
        "original_index": record.original_index,
        "text": record.text,
        "true_label": "" if record.true_label is None else record.true_label,
        "true_label_name": "" if record.true_label is None else LABEL_NAMES[record.true_label],
        "pred_label": record.pred_label,
        "pred_label_name": LABEL_NAMES[record.pred_label],
        "target_class": record.target_class,
        "target_class_name": LABEL_NAMES[record.target_class],
        "outcome": record.outcome,
        "neg_prob": record.neg_prob,
        "pos_prob": record.pos_prob,
        "confidence": record.confidence,
        "original_len": record.original_len,
        "tokens": " ".join(record.tokens[: record.original_len]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write rows to CSV with Excel-friendly UTF-8 BOM.

    `utf-8-sig`를 쓰면 Windows Excel에서 한국어 column/text가 깨질 가능성이 낮다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any, digits: int = 4) -> str:
    """Format numeric values for markdown tables without crashing on strings."""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Create a compact GitHub-flavored markdown table."""
    if not rows:
        return "\n없음\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep] + body)
