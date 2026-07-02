"""Filter activation analysis for TextCNN.

Filter activation은 CNN 내부를 직접 들여다보는 분석이다. 각 convolution filter가
문장 안의 어떤 n-gram window에서 가장 크게 반응했는지 찾고, 마지막 classifier의
weight까지 함께 보아 그 filter가 긍정/부정 중 어느 방향에 가까운지 해석한다.
"""

from __future__ import annotations

from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, get_conv_activations
from XAI.shared.schemas import LABEL_NAMES, SampleRecord


def compute_filter_class_direction(model: CNN_Sentiment) -> list[dict[str, Any]]:
    """Compute whether each pooled filter feature leans positive or negative.

    CNN의 conv filter activation은 "패턴을 감지했다"는 뜻일 뿐, 그 패턴이 긍정인지
    부정인지는 마지막 fully-connected layer weight가 결정한다.

    positive_direction = fc.weight[positive] - fc.weight[negative]

    이 값이 양수면 해당 feature가 커질수록 긍정 logit이 상대적으로 더 커지고,
    음수면 부정 쪽으로 더 기여한다고 볼 수 있다.
    """
    rows: list[dict[str, Any]] = []
    fc_weight = model.fc.weight.detach().cpu()
    for block_idx, filter_size in enumerate(model.filter_sizes):
        for filter_idx in range(model.n_filters):
            # concat 순서가 [filter_size=3 features, filter_size=4 features, ...]이므로
            # block index와 filter index를 fc 입력 feature index로 변환한다.
            feature_idx = block_idx * model.n_filters + filter_idx
            positive_direction = float(fc_weight[1, feature_idx] - fc_weight[0, feature_idx])
            rows.append(
                {
                    "filter_size": filter_size,
                    "conv_block_idx": block_idx,
                    "filter_idx": filter_idx,
                    "feature_idx": feature_idx,
                    "negative_weight": float(fc_weight[0, feature_idx]),
                    "positive_weight": float(fc_weight[1, feature_idx]),
                    "positive_direction": positive_direction,
                    "direction_label": "positive" if positive_direction >= 0 else "negative",
                }
            )
    return rows


def run_filter_activation_analysis(
    model: CNN_Sentiment, samples: list[SampleRecord], device: torch.device
) -> list[dict[str, Any]]:
    """Find the strongest n-gram activation for every filter and sample.

    Output row 하나는 다음 뜻을 가진다.
        "이 sample에서 filter_size=N의 filter_idx=K는 position=P의 n-gram에
        activation=A로 가장 강하게 반응했고, target class에는 contribution=C만큼
        기여했다."
    """
    rows: list[dict[str, Any]] = []
    fc_weight = model.fc.weight.detach().cpu()
    for record in samples:
        activations = get_conv_activations(model, record.ids, device)
        for block_idx, conved in enumerate(activations):
            filter_size = model.filter_sizes[block_idx]

            # padding 위치까지 포함하면 "<pad> <pad> ..." 같은 window가 top activation에
            # 섞일 수 있다. 실제 token으로 만들 수 있는 window만 분석한다.
            valid_positions = max(0, record.original_len - filter_size + 1)
            if valid_positions == 0:
                continue
            conved = conved[0, :, :valid_positions]

            # 각 filter별로 문장 안에서 가장 activation이 큰 위치를 찾는다.
            # values/indices shape: [n_filters]
            values, indices = torch.max(conved, dim=1)
            for filter_idx in range(model.n_filters):
                pos = int(indices[filter_idx].item())
                activation = float(values[filter_idx].item())
                feature_idx = block_idx * model.n_filters + filter_idx
                target_weight = float(fc_weight[record.target_class, feature_idx])
                negative_weight = float(fc_weight[0, feature_idx])
                positive_weight = float(fc_weight[1, feature_idx])
                positive_direction = positive_weight - negative_weight
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
                        "conv_block_idx": block_idx,
                        "filter_idx": filter_idx,
                        "feature_idx": feature_idx,
                        "activation": activation,
                        "position": pos,
                        "ngram": " ".join(record.tokens[pos : pos + filter_size]),
                        "negative_weight": negative_weight,
                        "positive_weight": positive_weight,
                        "positive_direction": positive_direction,
                        "direction_label": "positive" if positive_direction >= 0 else "negative",
                        "target_fc_weight": target_weight,

                        # contribution은 "이 sample의 activation 크기"와
                        # "target class classifier weight"를 곱한 값이다.
                        "target_contribution": activation * target_weight,
                    }
                )
    return rows
