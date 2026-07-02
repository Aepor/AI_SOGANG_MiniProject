"""n-gram occlusion for TextCNN sentiment explanations.

CNN은 filter size 3/4/5로 연속 token window를 본다. 따라서 단어 하나보다
"너무 지루하다", "시간 아깝다" 같은 n-gram을 가렸을 때의 변화가 CNN 구조를 더
잘 설명할 수 있다.
"""

from __future__ import annotations

from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, predict_batch_ids, predict_one
from XAI.CNN.shared.schemas import LABEL_NAMES, SampleRecord


def run_ngram_occlusion(
    model: CNN_Sentiment,
    samples: list[SampleRecord],
    pad_idx: int,
    ngram_sizes: list[int],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Compute n-gram occlusion scores for each sample.

    `ngram_sizes`에는 보통 1,2,3,4,5를 넣는다. 3/4/5는 CNN filter size와 직접
    대응하고, 1/2는 비교 기준선 역할을 한다.
    """
    rows: list[dict[str, Any]] = []
    for record in samples:
        # 원본 target 확률/logit이 모든 masked window의 비교 기준이다.
        base = predict_one(model, record.ids, device, batch_size)
        base_prob = float(base["probs"][record.target_class].item())
        base_logit = float(base["logits"][record.target_class].item())
        masked_ids = []
        meta = []
        for n in ngram_sizes:
            if n <= 0 or n > record.original_len:
                continue
            for start in range(0, record.original_len - n + 1):
                end = start + n
                ids = list(record.ids)

                # window 내부 token을 모두 <pad>로 바꾼다. 길이는 유지되므로 CNN filter가
                # 보는 position structure는 그대로이고, 해당 phrase 정보만 사라진다.
                for pos in range(start, end):
                    ids[pos] = pad_idx
                masked_ids.append(ids)
                meta.append((n, start, end, " ".join(record.tokens[start:end])))

        # 가능한 모든 n-gram window를 batch로 예측한다.
        logits, probs = predict_batch_ids(model, masked_ids, device, batch_size)
        for (n, start, end, ngram), masked_logit, masked_prob in zip(meta, logits, probs):
            target_prob = float(masked_prob[record.target_class].item())
            target_logit = float(masked_logit[record.target_class].item())
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "text": record.text,
                    "true_label": "" if record.true_label is None else record.true_label,
                    "pred_class": record.pred_label,
                    "target_class": record.target_class,
                    "target_class_name": LABEL_NAMES[record.target_class],
                    "n": n,
                    "start_pos": start,
                    "end_pos": end,
                    "ngram": ngram,
                    "base_prob": base_prob,
                    "base_logit": base_logit,
                    "masked_prob": target_prob,
                    "masked_logit": target_logit,
                    "prob_drop": base_prob - target_prob,
                    "logit_drop": base_logit - target_logit,
                }
            )
    return rows


def extract_topk_ngram(rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Return the strongest n-gram evidence rows by probability drop."""
    return sorted(rows, key=lambda row: float(row["prob_drop"]), reverse=True)[:top_k]
