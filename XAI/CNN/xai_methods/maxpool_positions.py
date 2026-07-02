"""Max-pooling position analysis for TextCNN.

TextCNN은 각 filter가 모든 n-gram 위치를 훑은 뒤 global max-pooling으로 가장 큰
activation 위치 하나만 남긴다. 이 파일은 그 "선택된 위치"를 token n-gram으로 복원해
모델이 최종 feature로 어떤 구절을 가져갔는지 보여준다.
"""

from __future__ import annotations

from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, get_conv_activations
from XAI.shared.schemas import LABEL_NAMES, SampleRecord


def run_maxpool_position_analysis(
    model: CNN_Sentiment, samples: list[SampleRecord], device: torch.device
) -> list[dict[str, Any]]:
    """Recover max-pooling positions and their target-class contributions."""
    rows: list[dict[str, Any]] = []
    fc_weight = model.fc.weight.detach().cpu()
    for record in samples:
        activations = get_conv_activations(model, record.ids, device)
        for block_idx, conved in enumerate(activations):
            filter_size = model.filter_sizes[block_idx]

            # 실제 token에서 시작하는 n-gram만 해석한다. padding window는 설명에서 제외한다.
            valid_positions = max(0, record.original_len - filter_size + 1)
            if valid_positions == 0:
                continue
            conved = conved[0, :, :valid_positions]

            # values는 max activation 값, indices는 그 activation이 나온 token 시작 위치이다.
            values, indices = torch.max(conved, dim=1)
            for filter_idx in range(model.n_filters):
                pos = int(indices[filter_idx].item())
                activation = float(values[filter_idx].item())
                feature_idx = block_idx * model.n_filters + filter_idx
                target_weight = float(fc_weight[record.target_class, feature_idx])
                rows.append(
                    {
                        "sample_id": record.sample_id,
                        "source": record.source,
                        "text": record.text,
                        "true_label": "" if record.true_label is None else record.true_label,
                        "pred_label": record.pred_label,
                        "target_class": record.target_class,
                        "target_class_name": LABEL_NAMES[record.target_class],
                        "filter_size": filter_size,
                        "filter_idx": filter_idx,

                        # feature_idx는 max-pooled feature가 fc layer에서 몇 번째 입력인지 뜻한다.
                        "feature_idx": feature_idx,
                        "max_position": pos,
                        "selected_ngram": " ".join(record.tokens[pos : pos + filter_size]),
                        "activation": activation,
                        "target_fc_weight": target_weight,

                        # target contribution이 클수록 해당 selected_ngram이 target class logit에
                        # 더 크게 들어갔다고 볼 수 있다.
                        "target_contribution": activation * target_weight,
                    }
                )
    return rows
