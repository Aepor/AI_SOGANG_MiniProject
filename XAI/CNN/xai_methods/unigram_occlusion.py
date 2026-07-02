"""Unigram occlusion for TextCNN sentiment explanations.

Occlusion은 "입력 일부를 가렸을 때 target class 확률이 얼마나 변하는가"를 보는
가장 직관적인 XAI 방법이다. 이 파일은 token 하나씩 `<pad>`로 바꿔 보며
각 token의 중요도를 측정한다.
"""

from __future__ import annotations

from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, predict_batch_ids, predict_one
from XAI.shared.schemas import LABEL_NAMES, SampleRecord


def run_unigram_occlusion(
    model: CNN_Sentiment,
    samples: list[SampleRecord],
    pad_idx: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Compute one-token occlusion scores for each sample.

    Score definition:
        prob_drop = P(target | original) - P(target | token_i masked)

    prob_drop이 양수면 해당 token을 가렸을 때 target 확률이 떨어진 것이다. 즉,
    모델이 그 token을 target class 판단 근거로 사용했다고 해석할 수 있다.
    """
    rows: list[dict[str, Any]] = []
    for record in samples:
        # 먼저 원본 문장에 대한 target class 확률/logit을 기준값으로 저장한다.
        base = predict_one(model, record.ids, device, batch_size)
        base_prob = float(base["probs"][record.target_class].item())
        base_logit = float(base["logits"][record.target_class].item())
        masked_ids = []
        meta = []

        # 실제 token 위치만 가린다. padding 위치는 원래 입력 정보가 아니므로 제외한다.
        for pos in range(record.original_len):
            ids = list(record.ids)

            # 삭제가 아니라 <pad> 치환을 사용한다. 삭제하면 뒤 token 위치가 당겨져서
            # "그 token의 영향"과 "위치 이동의 영향"이 섞이기 때문이다.
            ids[pos] = pad_idx
            masked_ids.append(ids)
            meta.append((pos, record.tokens[pos]))

        # 한 문장에 대해 token 수만큼 생긴 masked input을 batch로 한 번에 예측한다.
        logits, probs = predict_batch_ids(model, masked_ids, device, batch_size)
        for (pos, token), masked_logit, masked_prob in zip(meta, logits, probs):
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
                    "base_prob": base_prob,
                    "base_logit": base_logit,
                    "position": pos,
                    "token": token,
                    "masked_prob": target_prob,
                    "masked_logit": target_logit,
                    "prob_drop": base_prob - target_prob,
                    "logit_drop": base_logit - target_logit,
                }
            )
    return rows


def extract_topk_unigram(rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Return the most influential unigram rows by probability drop."""
    return sorted(rows, key=lambda row: float(row["prob_drop"]), reverse=True)[:top_k]
