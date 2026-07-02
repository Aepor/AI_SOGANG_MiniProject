"""Integrated Gradients attribution for TextCNN.

Integrated Gradients(IG)는 baseline 입력에서 실제 입력까지 천천히 이동하면서
target logit의 gradient를 누적하는 XAI 방법이다. Gradient x Input이 현재 입력 한
점에서의 민감도만 보는 반면, IG는 baseline -> input 경로 전체를 평균내기 때문에
더 안정적인 attribution을 기대할 수 있다.

텍스트 모델에서는 token id 자체가 이산값이라 직접 보간할 수 없다. 그래서 이 구현은
embedding space에서 다음 경로를 사용한다.

    baseline_embedding = embedding(<pad>, <pad>, ..., <pad>)
    input_embedding    = embedding(token_1, token_2, ..., <pad>)
    path(alpha)        = baseline + alpha * (input - baseline)
"""

from __future__ import annotations

from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, forward_from_embedded_for_xai
from XAI.shared.schemas import LABEL_NAMES, SampleRecord


def _integrated_gradients_for_one_sample(
    model: CNN_Sentiment,
    ids: list[int],
    target_class: int,
    pad_idx: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Return token-level signed IG scores for one encoded sample.

    Args:
        model: eval mode TextCNN.
        ids: padded token id sequence, length = max_len.
        target_class: class logit to explain.
        pad_idx: `<pad>` vocabulary id used for the all-pad baseline.
        steps: number of integration intervals. Larger is more accurate but slower.
        device: CPU/CUDA device.

    Returns:
        Tensor of shape `[seq_len]`. Each value is the embedding-dimension sum of
        integrated attribution for that token position.
    """
    if steps <= 0:
        raise ValueError("Integrated Gradients steps must be a positive integer.")

    ids_tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    baseline_ids = torch.full_like(ids_tensor, fill_value=pad_idx)

    # input_emb/baseline_emb shape: [1, seq_len, embedding_dim].
    input_emb = model.embedding(ids_tensor).detach()
    baseline_emb = model.embedding(baseline_ids).detach()
    diff = input_emb - baseline_emb

    # Trapezoidal rule: use steps+1 points from 0 to 1, then weight endpoints by 1/2.
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device).view(-1, 1, 1)
    interpolated = baseline_emb + alphas * diff
    interpolated.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = forward_from_embedded_for_xai(model, interpolated)
    target_sum = logits[:, target_class].sum()
    target_sum.backward()

    grads = interpolated.grad.detach()
    if steps == 1:
        avg_grads = grads.mean(dim=0, keepdim=True)
    else:
        # Average gradient along the path with trapezoidal endpoint weighting.
        weighted_sum = grads[0] + grads[-1] + 2.0 * grads[1:-1].sum(dim=0)
        avg_grads = (weighted_sum / (2.0 * steps)).unsqueeze(0)

    # IG attribution = (input - baseline) * average_gradient.
    attributions = diff * avg_grads
    return attributions.sum(dim=2).squeeze(0).detach().cpu()


def run_integrated_gradients(
    model: CNN_Sentiment,
    samples: list[SampleRecord],
    pad_idx: int,
    device: torch.device,
    steps: int = 32,
) -> list[dict[str, Any]]:
    """Compute token-level Integrated Gradients attribution for all samples.

    Score columns mirror `cnn_gradient_x_input.csv` so the two methods are easy
    to compare:

        signed_score: target class 방향 attribution. 양수면 target logit을 올리는 방향.
        abs_score: 방향을 무시한 중요도 크기.
        normalized_abs_score: sample 안에서 0~1로 정규화한 중요도.
    """
    rows: list[dict[str, Any]] = []
    model.eval()
    for record in samples:
        scores = _integrated_gradients_for_one_sample(
            model=model,
            ids=record.ids,
            target_class=record.target_class,
            pad_idx=pad_idx,
            steps=steps,
            device=device,
        )
        abs_scores = scores.abs()
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
                    "ig_steps": steps,
                    "baseline": "all_pad",
                }
            )
    return rows


def extract_topk_integrated_gradients(
    rows: list[dict[str, Any]], top_k: int = 5
) -> list[dict[str, Any]]:
    """Return the strongest IG rows by normalized absolute score."""
    return sorted(rows, key=lambda row: float(row["normalized_abs_score"]), reverse=True)[:top_k]
