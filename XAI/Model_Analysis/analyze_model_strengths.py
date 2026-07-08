"""XAI output JSON을 이용해 모델별 설명 특성을 비교한다.

이 스크립트는 XAI/outputs_json/output_*.json을 읽고, 예측 확률과 단어별
XAI score를 기반으로 모델이 어떤 설명 부문에서 강한지 요약한다.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT_DIR = Path(__file__).resolve().parents[2]
XAI_DIR = ROOT_DIR / "XAI"
INPUT_DIR = XAI_DIR / "outputs_json"
OUTPUT_DIR = XAI_DIR / "Model_Analysis" / "outputs"

EPS = 1e-12
TOP_K = 3
COVERAGE_THRESHOLD = 0.02

MODEL_LABELS = {
    "cnn": "CNN",
    "fnn": "FNN",
    "transformer": "Transformer",
}

METHOD_LABELS = {
    "attention": "Attention",
    "filter_activation": "Filter Activation",
    "ig": "Integrated Gradients",
    "ig_50": "Integrated Gradients 50 steps",
    "ig_100": "Integrated Gradients 100 steps",
    "occlusion": "Occlusion",
    "ngram_occlusion": "N-gram Occlusion",
    "unigram_occlusion": "Unigram Occlusion",
    "lime": "LIME",
}

REFERENCE_ONLY_METHODS = {"attention", "filter_activation", "lime"}
MODEL_COMPARISON_METHODS = {"ig_50", "ig_100", "occlusion", "unigram_occlusion"}

METRIC_INFO = [
    {
        "key": "alignment",
        "category": "Attribution Sign Agreement",
        "meaning": "예측 감성과 같은 방향의 attribution score가 전체 설명량에서 차지하는 비율입니다.",
        "ranking_rule": "예측 방향과 같은 부호의 설명량 비율이 높은 모델을 우수 모델로 봅니다.",
    },
    {
        "key": "focus",
        "category": "Max Attribution Share",
        "meaning": "가장 큰 절대 score를 가진 단어 하나가 전체 설명량에서 차지하는 비율입니다.",
        "ranking_rule": "핵심 단어를 더 뚜렷하게 잡아내는 모델을 우수 모델로 봅니다.",
    },
    {
        "key": "coverage",
        "category": "Attribution Coverage",
        "meaning": "문장 내 단어 중 전체 설명량의 2% 이상을 차지한 단어의 비율입니다.",
        "ranking_rule": "의미 있는 크기의 설명 신호를 더 많은 단어에 부여한 모델을 우수 모델로 봅니다.",
    },
    {
        "key": "top3_share",
        "category": "Top-3 Attribution Mass",
        "meaning": "절대 score 기준 상위 3개 단어가 전체 설명량에서 차지하는 비율입니다.",
        "ranking_rule": "중요 단어 3개에 설명이 선명하게 모이는 모델을 우수 모델로 봅니다.",
    },
    {
        "key": "method_agreement",
        "category": "Explanation Agreement",
        "meaning": "같은 모델 안에서 여러 XAI 방법이 비슷한 핵심 단어를 고르는 정도입니다.",
        "ranking_rule": "Top-3 단어 겹침과 절대 score 순위 상관이 높은 모델을 우수 모델로 봅니다.",
    },
]

METRIC_INFO_BY_KEY = {item["key"]: item for item in METRIC_INFO}

METRIC_MATH = {
    "alignment": {
        "formula": "alignment_i = sum(|s_ij| for sign(s_ij)=direction(y_hat_i)) / sum_j |s_ij|",
        "basis": "signed attribution의 부호가 예측 감성 방향과 일치하는 비율을 봅니다.",
        "notes": "positive 예측은 양수 score, negative 예측은 음수 score를 예측을 지지하는 설명으로 계산합니다.",
    },
    "focus": {
        "formula": "focus_i = max_j |s_ij| / sum_j |s_ij|",
        "basis": "L1 정규화된 attribution에서 가장 큰 단일 단어의 비중을 보는 방식입니다.",
        "notes": "값이 높을수록 한 핵심 단어에 설명이 선명하게 집중됩니다.",
    },
    "coverage": {
        "formula": "coverage_i = count_j(|s_ij| / sum_k |s_ik| >= 0.02) / n_i",
        "basis": "L1 정규화된 attribution에서 최소 비중 이상을 가진 단어만 세는 방식입니다.",
        "notes": "아주 작은 noise성 score를 제외하고, 전체 설명량의 2% 이상을 차지한 단어만 유효 설명 단어로 봅니다.",
    },
    "top3_share": {
        "formula": "top3_share_i = sum(top3_j |s_ij|) / sum_j |s_ij|, only if n_i > 3",
        "basis": "상위 k개 중요 특성의 attribution mass를 보는 feature concentration 방식입니다.",
        "notes": f"현재 k={TOP_K}로 설정되어 있으며, 단어 수가 {TOP_K}개 이하인 문장은 평균에서 제외합니다.",
    },
    "method_agreement": {
        "formula": "method_agreement = mean(top3_jaccard, spearman_abs_rank), only if n_i > 3",
        "basis": "Top-k 집합 유사도(Jaccard)와 절대 score 순위 상관(Spearman)을 결합합니다.",
        "notes": f"Top-{TOP_K} 비교가 전체 단어 비교가 되지 않도록 단어 수가 {TOP_K}개 이하인 문장은 제외합니다.",
    },
}


def parse_output_name(path: Path) -> tuple[str, str]:
    match = re.match(r"output_(?P<model>.+?)_(?P<method>.+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unsupported output filename: {path.name}")
    return match.group("model"), match.group("method")


def score_key(record: dict) -> str:
    for key in ("scores", "ig_scores", "occlusion_scores", "lime_scores", "attention_scores"):
        if key in record:
            return key
    raise KeyError(f"No score field found in record keys: {list(record.keys())}")


def safe_float_list(values: list) -> list[float]:
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def prediction_direction(prediction: str) -> int:
    return 1 if str(prediction).lower() == "positive" else -1


def top_indices(values: list[float], k: int = TOP_K) -> set[int]:
    if not values:
        return set()
    k = min(k, len(values))
    return {idx for idx, _ in sorted(enumerate(values), key=lambda item: abs(item[1]), reverse=True)[:k]}


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0 + 1.0
        for idx, _ in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_den = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_den = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_den < EPS or right_den < EPS:
        return math.nan
    return numerator / (left_den * right_den)


def spearman_abs(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    return pearson(rankdata([abs(value) for value in left]), rankdata([abs(value) for value in right]))


def mean_available(values: list[float]) -> float | None:
    valid_values = [
        value
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    return mean(valid_values) if valid_values else None


def is_available(value: float | None) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def record_metrics(record: dict, scores: list[float]) -> dict[str, float]:
    words = record.get("words", [])
    usable_len = min(len(words), len(scores))
    scores = scores[:usable_len]
    abs_scores = [abs(value) for value in scores]
    abs_total = sum(abs_scores)
    direction = prediction_direction(record.get("prediction", "positive"))

    if usable_len == 0 or abs_total < EPS:
        return {
            "confidence": float(record.get("probability", 0.0)),
            "alignment": 0.0,
            "focus": 0.0,
            "coverage": 0.0,
            "top3_share": math.nan,
            "avg_abs_score": 0.0,
            "word_count": float(len(words)),
        }

    aligned_mass = sum(abs(value) for value in scores if direction * value > 0)
    effective_count = sum(1 for value in scores if abs(value) / abs_total >= COVERAGE_THRESHOLD)
    top3_share = (
        sum(sorted(abs_scores, reverse=True)[:TOP_K]) / abs_total
        if usable_len > TOP_K
        else math.nan
    )

    return {
        "confidence": float(record.get("probability", 0.0)),
        "alignment": aligned_mass / abs_total,
        "focus": max(abs_scores) / abs_total,
        "coverage": effective_count / usable_len,
        "top3_share": top3_share,
        "avg_abs_score": abs_total / usable_len,
        "word_count": float(len(words)),
    }


def load_records() -> list[dict]:
    rows = []
    paths = sorted(INPUT_DIR.glob("output_*.json"))
    path_names = {path.name for path in paths}
    if {"output_fnn_ig_50.json", "output_fnn_ig_100.json"} & path_names:
        paths = [path for path in paths if path.name != "output_fnn_ig.json"]

    for path in paths:
        model, raw_method = parse_output_name(path)
        records = json.loads(path.read_text(encoding="utf-8"))
        for idx, record in enumerate(records, start=1):
            key = score_key(record)
            scores = safe_float_list(record.get(key, []))
            metrics = record_metrics(record, scores)
            rows.append(
                {
                    "model": model,
                    "method": raw_method,
                    "source_method": raw_method,
                    "source_methods": [raw_method],
                    "case_id": idx,
                    "text": record.get("text", ""),
                    "prediction": record.get("prediction", ""),
                    "probability": float(record.get("probability", 0.0)),
                    "words": record.get("words", []),
                    "score_key": key,
                    "scores": scores,
                    **metrics,
                }
            )
    return rows


def model_comparison_rows(rows: list[dict]) -> list[dict]:
    """Build rows for model-level comparison.

    CNN has both unigram and n-gram occlusion in the method profile. For the
    model-level comparison, unigram occlusion is used as CNN's representative
    Occlusion result so each model contributes one Occlusion method.
    """
    comparison_rows = []
    for row in rows:
        if row["method"] in REFERENCE_ONLY_METHODS:
            continue
        if row["method"] not in MODEL_COMPARISON_METHODS:
            continue

        comparison_row = dict(row)
        if row["method"] == "unigram_occlusion":
            comparison_row["method"] = "occlusion"
        comparison_rows.append(comparison_row)
    return comparison_rows


def aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    metric_keys = ("alignment", "focus", "coverage", "top3_share", "avg_abs_score", "word_count")
    summaries = []
    for group_values, group_rows in sorted(grouped.items()):
        summary = {key: value for key, value in zip(group_keys, group_values)}
        summary["case_count"] = len(group_rows)
        source_methods = sorted({source for row in group_rows for source in row.get("source_methods", [row.get("source_method", row["method"])])})
        summary["source_methods"] = source_methods
        for metric in metric_keys:
            values = [row[metric] for row in group_rows]
            summary[metric] = mean_available(values)
            if metric == "top3_share":
                summary["top3_share_case_count"] = sum(1 for value in values if is_available(value))
        summaries.append(summary)
    return summaries


def common_methods(rows: list[dict]) -> set[str]:
    methods_by_model = defaultdict(set)
    for row in rows:
        methods_by_model[row["model"]].add(row["method"])
    if not methods_by_model:
        return set()
    return set.intersection(*methods_by_model.values()) - REFERENCE_ONLY_METHODS


def filter_common_method_rows(rows: list[dict]) -> tuple[list[dict], set[str]]:
    methods = common_methods(rows)
    return [row for row in rows if row["method"] in methods], methods


def filter_common_text_rows(rows: list[dict], methods: set[str]) -> tuple[list[dict], list[str]]:
    models = sorted({row["model"] for row in rows})
    required_pairs = {(model, method) for model in models for method in methods}
    pairs_by_text = defaultdict(set)
    for row in rows:
        if row["method"] in methods:
            pairs_by_text[row["text"]].add((row["model"], row["method"]))

    common_texts = sorted(
        text
        for text, pairs in pairs_by_text.items()
        if required_pairs.issubset(pairs)
    )
    common_text_index = {text: idx for idx, text in enumerate(common_texts, start=1)}
    filtered_rows = []
    for row in rows:
        if row["method"] not in methods or row["text"] not in common_text_index:
            continue
        filtered_row = dict(row)
        filtered_row["case_id"] = common_text_index[row["text"]]
        filtered_rows.append(filtered_row)
    return filtered_rows, common_texts


def method_agreement(rows: list[dict]) -> list[dict]:
    by_model_case = defaultdict(list)
    for row in rows:
        by_model_case[(row["model"], row["text"])].append(row)

    pair_rows = []
    for (model, case_id), case_rows in sorted(by_model_case.items()):
        for left, right in itertools.combinations(sorted(case_rows, key=lambda row: row["method"]), 2):
            if len(left["scores"]) != len(right["scores"]):
                continue
            if len(left["scores"]) <= TOP_K:
                continue
            pair_rows.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "method_pair": f"{left['method']} vs {right['method']}",
                    "top3_jaccard": jaccard(top_indices(left["scores"]), top_indices(right["scores"])),
                    "spearman_abs": spearman_abs(left["scores"], right["scores"]),
                }
            )

    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[row["model"]].append(row)

    summaries = []
    for model, model_rows in sorted(grouped.items()):
        valid_spearman = [row["spearman_abs"] for row in model_rows if is_available(row["spearman_abs"])]
        top3_jaccard = mean_available([row["top3_jaccard"] for row in model_rows])
        spearman = mean_available(valid_spearman)
        agreement_parts = [value for value in [top3_jaccard, spearman] if is_available(value)]
        summaries.append(
            {
                "model": model,
                "pair_count": len(model_rows),
                "top3_jaccard": top3_jaccard,
                "spearman_abs": spearman,
                "method_agreement": mean(agreement_parts) if agreement_parts else None,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def display_value(value: float | None) -> float | str | None:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def format_metric(value: float | None) -> str:
    if not is_available(value):
        return "계산 제외"
    return f"{value:.4f}"


def localize_summary_rows(rows: list[dict]) -> list[dict]:
    localized = []
    for row in rows:
        output = {}
        if "model" in row:
            output["Model"] = model_label(row["model"])
        if "method" in row:
            output["XAI Method"] = method_label(row["method"])
        if "source_methods" in row:
            output["Source Method(s)"] = ", ".join(method_label(method) for method in row["source_methods"])
        if "case_count" in row:
            output["Case Count"] = row["case_count"]
        if "top3_share_case_count" in row:
            output["Top-3 Eligible Case Count"] = row["top3_share_case_count"]
        output.update(
            {
                "Attribution Sign Agreement": display_value(row.get("alignment")),
                "Max Attribution Share": display_value(row.get("focus")),
                "Attribution Coverage": display_value(row.get("coverage")),
                "Top-3 Attribution Mass": display_value(row.get("top3_share")),
                "Mean Absolute Score": display_value(row.get("avg_abs_score")),
                "Mean Word Count": display_value(row.get("word_count")),
            }
        )
        localized.append(output)
    return localized


def localize_agreement_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Model": model_label(row["model"]),
            "Method Pair Count": row["pair_count"],
            "Top-3 Jaccard Overlap": display_value(row["top3_jaccard"]),
            "Spearman Rank Correlation (Absolute Scores)": display_value(row["spearman_abs"]),
            "Explanation Agreement": display_value(row["method_agreement"]),
        }
        for row in rows
    ]


def localize_winner_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Evaluation Category": row["category"],
            "Metric": METRIC_INFO_BY_KEY.get(row["metric"], {}).get("category", row["metric"]),
            "Best Model": model_label(row["winner"]),
            "Score": row["score"],
        }
        for row in rows
    ]


def metric_winners(model_summary: list[dict], agreement_summary: list[dict]) -> list[dict]:
    agreement_by_model = {row["model"]: row for row in agreement_summary}
    merged = []
    for row in model_summary:
        agreement = agreement_by_model.get(row["model"], {})
        merged.append({**row, **agreement})

    winners = []
    for item in METRIC_INFO:
        metric = item["key"]
        candidates = [row for row in merged if metric in row and is_available(row[metric])]
        if not candidates:
            continue
        winner = max(candidates, key=lambda row: row[metric])
        winners.append(
            {
                "category": item["category"],
                "metric": metric,
                "winner": winner["model"],
                "score": winner[metric],
            }
        )
    return winners


def write_markdown(
    path: Path,
    winners: list[dict],
    model_summary: list[dict],
    agreement_summary: list[dict],
    common_method_set: set[str],
    common_text_count: int,
) -> None:
    agreement_by_model = {row["model"]: row for row in agreement_summary}
    common_methods_text = ", ".join(method_label(method) for method in sorted(common_method_set))
    lines = [
        "# 모델 강점 분석 리포트",
        "",
        "이 리포트는 `XAI/outputs_json`에 저장된 예측 확률과 단어별 XAI score를 사용해 모델별 설명 특성을 비교합니다.",
        "최종 모델 순위는 모든 모델에 공통으로 존재하는 XAI 방법만 사용해 계산합니다.",
        f"최종 모델 순위는 모든 모델에 공통으로 존재하는 입력 문장 **{common_text_count}개**만 사용해 계산합니다.",
        "현재 모델 비교에 사용된 공통 방법: "
        f"**{common_methods_text}**.",
        "",
        "주의: `outputs_json`에는 정답 라벨이 없으므로 이 분석은 분류 정확도 평가가 아닙니다. "
        "대신 모델의 설명 score 형태를 비교합니다.",
        "",
        "## 부문별 우수 모델",
        "",
    ]
    for row in winners:
        lines.append(f"- {row['category']}: **{model_label(row['winner'])}** ({row['score']:.4f})")

    lines.extend(["", "## 모델별 요약", ""])
    for row in model_summary:
        agreement = agreement_by_model.get(row["model"], {})
        lines.append(
            "- "
            f"**{model_label(row['model'])}**: Attribution Sign Agreement={row['alignment']:.4f}, "
            f"Max Attribution Share={row['focus']:.4f}, "
            f"Attribution Coverage={row['coverage']:.4f}, "
            f"Top-3 Attribution Mass={format_metric(row['top3_share'])}, "
            f"Explanation Agreement={format_metric(agreement.get('method_agreement'))}"
        )

    lines.extend(["", "## 순위 산정 방법", ""])
    for item in METRIC_INFO:
        lines.append(f"- **{item['category']}**: {item['meaning']} {item['ranking_rule']}")

    lines.extend(
        [
            "",
            "## 계산 방식 요약",
            "",
            "- 단어별 score는 모델과 XAI 방법마다 크기 범위가 다를 수 있으므로, 문장 단위에서 절대값 합을 기준으로 비율형 지표를 계산합니다.",
            f"- `Attribution Coverage`는 정규화된 절대 score가 {COVERAGE_THRESHOLD:.2f} 이상인 단어만 유효 설명 단어로 계산합니다.",
            f"- `Top-3 Attribution Mass`와 `Top-3 Jaccard Overlap`은 단어 수가 {TOP_K + 1}개 이상인 문장에 대해서만 계산합니다.",
            "- `Attribution Sign Agreement`는 positive 예측이면 양수 score, negative 예측이면 음수 score를 예측을 지지하는 설명으로 봅니다.",
            "- `Explanation Agreement`는 같은 문장에 대해 방법별 상위 3개 단어 겹침도와 절대 score 순위 상관을 평균낸 값입니다.",
            "- FNN에만 있는 LIME처럼 특정 모델에만 존재하는 방법은 방법별 참고표에는 남기지만, 최종 모델 순위에는 포함하지 않습니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_methodology(path: Path, common_method_set: set[str], common_text_count: int) -> None:
    common_methods_text = ", ".join(method_label(method) for method in sorted(common_method_set))
    lines = [
        "# 모델 분석 방법론",
        "",
        "## 분석 목적",
        "",
        "이 프로그램은 `outputs_json`에 들어 있는 모델별 XAI 결과를 이용해 어떤 모델이 어떤 설명 특성에서 강한지 비교합니다.",
        "정답 라벨이 포함되어 있지 않기 때문에 정확도, 재현율, F1-score 같은 분류 성능을 계산하지 않습니다.",
        "",
        "## 비교 범위",
        "",
        f"- 최종 모델 비교에는 모든 모델에 공통으로 존재하는 방법만 사용합니다: **{common_methods_text}**.",
        f"- 최종 모델 비교에는 모든 모델과 공통 XAI 방법에 모두 존재하는 입력 문장만 사용합니다. 현재 공통 입력 문장 수는 **{common_text_count}개**입니다.",
        "- 특정 모델에만 있는 방법은 `method_strength_summary.csv`에서 참고용으로 확인합니다.",
        "- 예를 들어 FNN의 LIME은 FNN 내부 설명 특성을 보는 데는 유용하지만, Transformer에 대응 방법이 없으면 최종 모델 순위에는 넣지 않습니다.",
        f"- Top-3 기반 지표는 단어 수가 {TOP_K + 1}개 이상인 문장에 대해서만 계산합니다.",
        "",
        "## 수식 기호",
        "",
        "- `i`: 문장 또는 사례 인덱스입니다.",
        "- `j`: 문장 안의 단어 인덱스입니다.",
        "- `s_ij`: i번째 문장의 j번째 단어에 부여된 XAI score입니다.",
        "- `|s_ij|`: score의 절대값이며, 설명량의 크기로 사용합니다.",
        "- `p_i(y_hat_i)`: i번째 문장에서 모델이 예측 클래스에 부여한 확률입니다.",
        "- `direction(y_hat_i)`: 예측이 positive이면 양수, negative이면 음수 방향입니다.",
        "- 각 지표는 먼저 문장 단위로 계산한 뒤, 모델별 평균을 냅니다.",
        "",
        "## 지표별 순위 산정",
        "",
    ]
    for item in METRIC_INFO:
        math_info = METRIC_MATH[item["key"]]
        lines.extend(
            [
                f"### {item['category']}",
                "",
                f"- 의미: {item['meaning']}",
                f"- 순위 기준: {item['ranking_rule']}",
                f"- 간단 수식: `{math_info['formula']}`",
                f"- 차용한 방식: {math_info['basis']}",
                f"- 해석 메모: {math_info['notes']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 산출 파일 해석",
            "",
            "- `category_winners.csv`: 평가 부문별로 어느 모델이 가장 높은 점수를 받았는지 정리한 표입니다.",
            "- `model_strength_summary.csv`: 공통 XAI 방법만 사용해 모델별 평균 지표를 계산한 표입니다.",
            "- `method_strength_summary.csv`: 모델-방법 조합별 지표입니다. 공통 방법이 아닌 LIME도 참고용으로 포함됩니다.",
            "- `method_agreement_summary.csv`: 같은 모델 안에서 여러 XAI 방법이 얼마나 일관된 설명을 내는지 정리한 표입니다.",
            "- `model_strength_report.md`: 사람이 읽기 쉬운 최종 요약 보고서입니다.",
            "- `model_strength_report.json`: 위 내용을 프로그램에서 다시 활용하기 위한 구조화된 결과입니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def method_note(method: str, source_methods: list[str]) -> str:
    if method == "attention":
        return "모델 내부 attention 가중치를 보는 참고형 설명입니다. score가 대부분 양수라 방향성 지표는 보조적으로만 해석합니다."
    if method == "filter_activation":
        return "CNN 필터가 강하게 반응한 단어를 보는 모델 특화 설명입니다. 다른 모델과의 최종 비교에는 넣지 않습니다."
    if method == "lime":
        return "입력 perturbation으로 만든 local surrogate 설명입니다. FNN 내부 참고용으로 유지합니다."
    if method == "unigram_occlusion":
        return "단어를 하나씩 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다. 모델 간 최종 비교에서 CNN의 Occlusion 대표값으로 사용합니다."
    if method == "ngram_occlusion":
        return "연속된 여러 단어 묶음을 가렸을 때 예측 확률 변화를 보는 제거 기반 설명입니다. CNN 내부 방법 비교용으로 유지합니다."
    if method == "occlusion":
        return "단어를 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다."
    if method in {"ig_50", "ig_100", "ig"}:
        return "baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다."
    return "해당 모델의 XAI 방법별 score 특성을 요약한 결과입니다."


def best_method(rows: list[dict], metric: str) -> dict | None:
    candidates = [row for row in rows if is_available(row.get(metric))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[metric])


def write_xai_method_profile(path: Path, method_summary: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in method_summary:
        grouped[row["model"]].append(row)

    lines = [
        "# 모델별 XAI 방법 비교",
        "",
        "이 문서는 각 모델 내부에서 어떤 XAI 방법이 어떤 특성을 보이는지 정리합니다.",
        "`method_strength_summary.csv`를 사람이 읽기 쉽게 해석한 파일이며, 모델 간 최종 우수 모델 비교와는 목적이 다릅니다.",
        "",
        "## 해석 기준",
        "",
        "- Attribution Sign Agreement: 모델 예측 방향과 XAI score 부호가 얼마나 잘 맞는지 봅니다.",
        "- Attribution Coverage: 전체 설명량의 2% 이상을 받은 단어 비율입니다.",
        "- Top-3 Attribution Mass: 설명이 주요 단어 묶음에 얼마나 모이는지 보는 특성 지표입니다.",
        "",
    ]

    profile_metrics = [
        ("alignment", "Attribution Sign Agreement"),
        ("coverage", "Attribution Coverage"),
        ("top3_share", "Top-3 Attribution Mass"),
    ]

    for model, rows in sorted(grouped.items()):
        lines.extend([f"## {model_label(model)}", ""])

        lines.extend(["### 돋보이는 XAI 방법", ""])
        for metric, label in profile_metrics:
            winner = best_method(rows, metric)
            if winner is None:
                continue
            lines.append(
                f"- {label}: **{method_label(winner['method'])}** "
                f"({format_metric(winner[metric])})"
            )

        lines.extend(["", "### 방법별 특징", ""])
        for row in sorted(rows, key=lambda item: item["method"]):
            source_methods = row.get("source_methods", [row.get("method", "")])
            source_text = ", ".join(method_label(method) for method in source_methods)
            lines.extend(
                [
                    f"#### {method_label(row['method'])}",
                    "",
                    f"- 원본 방법: {source_text}",
                    f"- 사례 수: {row['case_count']}",
                    f"- Attribution Sign Agreement: {format_metric(row['alignment'])}",
                    f"- Attribution Coverage: {format_metric(row['coverage'])}",
                    f"- Top-3 Attribution Mass: {format_metric(row['top3_share'])}",
                    f"- 특징: {method_note(row['method'], source_methods)}",
                    "",
                ]
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_metric_naming_notes(path: Path) -> None:
    lines = [
        "# XAI Metric Naming Notes",
        "",
        "이 문서는 Model Analysis에서 쓰는 지표명이 기존 XAI 문헌의 표준 지표명인지,",
        "또는 프로젝트 비교 목적에 맞게 만든 커스텀 지표인지 구분합니다.",
        "",
        "## Literature-Aligned Names Used",
        "",
        "| Current Name | Why This Name Is Used | Standardness |",
        "|---|---|---|",
        "| Top-3 Jaccard Overlap | Top-k 집합 유사도를 Jaccard overlap으로 계산합니다. Jaccard는 표준 유사도 지표입니다. | Standard component |",
        "| Spearman Rank Correlation (Absolute Scores) | 절대 attribution score 순위의 상관을 Spearman correlation으로 계산합니다. Spearman은 표준 순위 상관 지표입니다. | Standard component |",
        "| Top-3 Attribution Mass | 상위 k개 feature가 전체 attribution mass에서 차지하는 비율입니다. top-k attribution/rationale 분석에서 흔한 표현입니다. | Literature-aligned, project-specific k=3 |",
        "",
        "## Project-Specific Metrics",
        "",
        "| Current Name | Computation | Why It Is Project-Specific |",
        "|---|---|---|",
        "| Attribution Sign Agreement | 예측이 positive이면 양수 attribution, negative이면 음수 attribution이 전체 절대 attribution에서 차지하는 비율입니다. | signed attribution을 예측 감성 방향과 맞춘 프로젝트 지표입니다. 기존 논문에서 이 이름으로 고정된 표준 지표는 아닙니다. |",
        "| Attribution Coverage | 전체 attribution mass의 2% 이상을 받은 단어의 비율입니다. | coverage/sparsity 계열 아이디어는 흔하지만 2% threshold와 단어 비율 계산은 프로젝트 설정입니다. |",
        "| Max Attribution Share | 가장 큰 단일 단어 attribution mass가 전체 attribution mass에서 차지하는 비율입니다. | sparsity/concentration 아이디어를 단어 하나 기준으로 단순화한 프로젝트 지표입니다. |",
        "| Top-3 Attribution Mass | 절대 score 기준 상위 3개 단어의 attribution mass 비율입니다. | top-k attribution mass라는 표현은 문헌 친화적이지만 k=3과 단어 단위 계산은 프로젝트 설정입니다. |",
        "| Explanation Agreement | Top-3 Jaccard Overlap과 Spearman Rank Correlation을 평균낸 값입니다. | 구성요소는 표준적이지만 둘을 단순 평균해 하나의 점수로 만든 것은 프로젝트 지표입니다. |",
        "",
        "## More Standard Alternatives Not Used Here",
        "",
        "- Comprehensiveness: 중요한 rationale을 제거했을 때 예측이 얼마나 변하는지 보는 ERASER 계열 지표입니다.",
        "- Sufficiency: rationale만 남겼을 때 예측이 얼마나 유지되는지 보는 ERASER 계열 지표입니다.",
        "- Infidelity: explanation과 입력 perturbation에 따른 출력 변화가 얼마나 맞는지 보는 지표입니다.",
        "- Sensitivity: 작은 입력 변화에 explanation이 얼마나 안정적인지 보는 지표입니다.",
        "- ROAR: 중요한 feature를 제거하고 재학습해 feature importance를 평가하는 benchmark 방식입니다.",
        "",
        "현재 프로젝트는 저장된 JSON의 단어별 score만 사용하므로, 재추론/제거 실험이 필요한 Comprehensiveness, Sufficiency, ROAR는 최종 지표로 쓰지 않았습니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_records()
    if not rows:
        raise SystemExit(f"No output_*.json files found in {INPUT_DIR}")

    method_summary = aggregate(rows, ("model", "method"))
    comparison_rows = model_comparison_rows(rows)
    common_rows, common_method_set = filter_common_method_rows(comparison_rows)
    if not common_rows:
        raise SystemExit("No common XAI methods found across models.")
    common_rows, common_texts = filter_common_text_rows(common_rows, common_method_set)
    if not common_rows:
        raise SystemExit("No common input texts found across models and common XAI methods.")

    model_summary = aggregate(common_rows, ("model",))
    agreement_summary = method_agreement(common_rows)
    winners = metric_winners(model_summary, agreement_summary)

    write_csv(OUTPUT_DIR / "method_strength_summary.csv", localize_summary_rows(method_summary))
    write_csv(OUTPUT_DIR / "model_strength_summary.csv", localize_summary_rows(model_summary))
    write_csv(OUTPUT_DIR / "method_agreement_summary.csv", localize_agreement_rows(agreement_summary))
    write_csv(OUTPUT_DIR / "category_winners.csv", localize_winner_rows(winners))
    write_xai_method_profile(OUTPUT_DIR / "xai_method_profile_by_model.md", method_summary)
    write_metric_naming_notes(OUTPUT_DIR / "metric_naming_notes.md")

    report = {
        "입력_폴더": str(INPUT_DIR),
        "모델_비교_범위": "모든 모델에 공통으로 존재하는 XAI 방법만 사용",
        "공통_XAI_방법": [method_label(method) for method in sorted(common_method_set)],
        "공통_입력_문장_수": len(common_texts),
        "공통_입력_문장": common_texts,
        "지표_설명": METRIC_INFO,
        "지표_수식": METRIC_MATH,
        "모델별_요약": localize_summary_rows(model_summary),
        "방법별_요약": localize_summary_rows(method_summary),
        "방법_일관성_요약": localize_agreement_rows(agreement_summary),
        "부문별_우수_모델": localize_winner_rows(winners),
        "모델별_XAI_방법_비교_파일": str(OUTPUT_DIR / "xai_method_profile_by_model.md"),
        "raw": {
            "common_methods": sorted(common_method_set),
            "common_text_count": len(common_texts),
            "common_texts": common_texts,
            "model_summary": model_summary,
            "method_summary": method_summary,
            "method_agreement": agreement_summary,
            "category_winners": winners,
        },
    }
    (OUTPUT_DIR / "model_strength_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        OUTPUT_DIR / "model_strength_report.md",
        winners,
        model_summary,
        agreement_summary,
        common_method_set,
        len(common_texts),
    )
    write_methodology(OUTPUT_DIR / "model_analysis_methodology.md", common_method_set, len(common_texts))

    print(f"Analyzed {len(rows)} model-method records from {INPUT_DIR}")
    print(f"Model-level comparison uses common methods only: {', '.join(sorted(common_method_set))}")
    print(f"Model-level comparison uses common input texts only: {len(common_texts)} texts")
    print(f"Saved analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
