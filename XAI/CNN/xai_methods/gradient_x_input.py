"""Gradient x Input attribution for TextCNN.

Gradient x Input은 target logit이 embedding 입력에 얼마나 민감한지 보는 방법이다.
Occlusion이 "가렸을 때 실제 확률 변화"를 보는 반면, 이 방법은 현재 입력 지점에서의
local sensitivity를 본다. CNN에서는 중요도가 인접 token으로 퍼질 수 있으므로
occlusion과 함께 보조 지표로 해석한다.
"""

from __future__ import annotations

from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, forward_from_embedded_for_xai
from XAI.shared.schemas import LABEL_NAMES, SampleRecord


def run_gradient_x_input(
    model: CNN_Sentiment, samples: list[SampleRecord], device: torch.device
) -> list[dict[str, Any]]:
    """Compute token-level Gradient x Input attribution.

    Score definition:
        signed_score_i = sum(gradient_i * embedding_i)

    signed score는 target class logit을 올리는 방향/내리는 방향을 모두 담고,
    abs score는 중요도 크기만 본다. figure와 top-k 비교에는 보통 abs score를 쓴다.
    """
    rows: list[dict[str, Any]] = []
    model.eval()
    for record in samples:
        ids = torch.LongTensor(record.ids).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)

        # 일반 forward는 내부에서 embedding을 만들기 때문에 embedding gradient를
        # 직접 잡기 어렵다. 여기서는 embedding tensor를 먼저 만든 뒤 retain_grad 한다.
        embedded = model.embedding(ids)
        embedded.retain_grad()

        # XAI용 forward는 embedding 이후부터 classifier까지 계산한다.
        logits = forward_from_embedded_for_xai(model, embedded)
        target_logit = logits[0, record.target_class]
        target_logit.backward()

        # grad/embedding shape: [seq_len, embedding_dim].
        # embedding_dim 방향으로 합산하면 token별 attribution 하나가 된다.
        grad = embedded.grad.detach()[0].cpu()
        emb = embedded.detach()[0].cpu()
        scores = (grad * emb).sum(dim=1)
        abs_scores = scores.abs()

        # padding 위치는 제외하고, 실제 token 중 최대값으로 0~1 정규화한다.
        max_abs = float(abs_scores[: record.original_len].max().item()) if record.original_len else 0.0
        for pos in range(record.original_len):
            signed = float(scores[pos].item())
            abs_score = float(abs_scores[pos].item())
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "text": record.text,
                    "true_label": "" if record.true_label is None else record.true_label,
                    "pred_label": record.pred_label,
                    "target_class": record.target_class,
                    "target_class_name": LABEL_NAMES[record.target_class],
                    "position": pos,
                    "token": record.tokens[pos],
                    "signed_score": signed,
                    "abs_score": abs_score,
                    "normalized_abs_score": 0.0 if max_abs == 0 else abs_score / max_abs,
                }
            )
    return rows
