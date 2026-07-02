"""TextCNN model and prediction helpers for XAI.

이 파일은 `nsmc_cnn.ipynb`의 `CNN_Sentiment` 구조를 그대로 옮긴 것이다.
XAI 분석에서 중요한 점은 모델 구조를 학습 때와 1픽셀도 다르게 만들지 않는 것이다.
layer 순서나 dropout 위치가 바뀌면 checkpoint를 제대로 해석할 수 없다.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Sentiment(nn.Module):
    """TextCNN sentiment classifier used by the NSMC CNN notebook.

    입력 shape:
        text: `[batch_size, seq_len]` 정수 token id.

    출력 shape:
        logits: `[batch_size, 2]`, class 0은 부정, class 1은 긍정.

    구조:
        embedding -> Conv2d(filter size 3/4/5) -> ReLU -> global max-pooling
        -> concat -> dropout -> linear classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_filters: int,
        filter_sizes: list[int],
        output_dim: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        # padding_idx를 지정하면 <pad> embedding은 학습 중 gradient update에서 제외된다.
        # XAI에서도 pad token을 "빈 자리"로 쓰기 때문에 이 설정이 중요하다.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # 텍스트 CNN이지만 PyTorch 구현은 Conv2d를 쓴다.
        # kernel_size=(fs, embedding_dim)이므로 세로 방향으로 fs개 token을 보고,
        # 가로 방향으로는 embedding vector 전체를 한 번에 덮는다.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        # filter size가 3개이고 각 size마다 n_filters개 feature가 나오므로
        # classifier 입력 차원은 len(filter_sizes) * n_filters이다.
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

        # XAI에서 filter size와 filter index를 다시 복원해야 해서 보관한다.
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Run the normal training/inference forward pass."""
        embedded = self.embedding(text)

        # Conv2d는 channel 차원이 필요하다. 텍스트 embedding sequence를
        # `[batch, 1, seq_len, embedding_dim]` 형태의 1-channel 이미지처럼 본다.
        embedded = embedded.unsqueeze(1)
        pooled = []
        for conv in self.convs:
            # conved shape: [batch, n_filters, seq_len - filter_size + 1]
            # 각 위치는 특정 n-gram window에 대한 filter activation이다.
            conved = F.relu(conv(embedded)).squeeze(3)

            # global max-pooling은 각 filter가 문장 전체에서 가장 강하게 반응한
            # 위치 하나만 최종 feature로 남긴다. maxpool XAI가 바로 이 위치를 분석한다.
            pool = torch.max(conved, dim=2)[0]
            pooled.append(pool)
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


def infer_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Infer model hyperparameters from a saved checkpoint.

    노트북이 별도 config 파일을 저장하지 않았기 때문에, embedding/filter weight shape에서
    vocab size, embedding dim, filter sizes 등을 역산한다.
    """
    embedding_shape = state_dict["embedding.weight"].shape
    conv_keys = sorted(
        [key for key in state_dict if key.startswith("convs.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )
    filter_sizes = [int(state_dict[key].shape[2]) for key in conv_keys]
    n_filters = int(state_dict[conv_keys[0]].shape[0])
    output_dim = int(state_dict["fc.weight"].shape[0])
    return {
        "vocab_size": int(embedding_shape[0]),
        "embedding_dim": int(embedding_shape[1]),
        "filter_sizes": filter_sizes,
        "n_filters": n_filters,
        "output_dim": output_dim,
        "dropout": 0.5,
    }


def predict_batch_ids(
    model: CNN_Sentiment,
    ids_batch: list[list[int]] | torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict logits/probabilities for many already-encoded samples.

    XAI는 같은 문장을 수십 번 masking해서 예측하므로 batch 처리 없이는 느리다.
    반환값은 CPU tensor로 돌려 CSV/후처리 코드에서 GPU 메모리를 붙잡지 않게 한다.
    """
    tensor = torch.LongTensor(ids_batch) if isinstance(ids_batch, list) else ids_batch.long()
    logits_parts = []
    probs_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor.size(0), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            logits_parts.append(logits.detach().cpu())
            probs_parts.append(probs.detach().cpu())
    return torch.cat(logits_parts, dim=0), torch.cat(probs_parts, dim=0)


def predict_one(
    model: CNN_Sentiment, ids: list[int], device: torch.device, batch_size: int
) -> dict[str, Any]:
    """Predict one encoded review and return the fields used by reports."""
    logits, probs = predict_batch_ids(model, [ids], device, batch_size)
    prob = probs[0]
    pred = int(torch.argmax(prob).item())
    return {
        "logits": logits[0],
        "probs": prob,
        "pred": pred,
        "neg_prob": float(prob[0].item()),
        "pos_prob": float(prob[1].item()),
        "confidence": float(prob[pred].item()),
    }


def get_conv_activations(
    model: CNN_Sentiment, ids: list[int], device: torch.device
) -> list[torch.Tensor]:
    """Return ReLU convolution activations for one encoded review.

    반환 list의 각 원소는 filter size 하나에 해당한다.
    shape은 `[1, n_filters, valid_positions]`이고, valid_positions는
    `seq_len - filter_size + 1`이다.
    """
    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    with torch.no_grad():
        embedded = model.embedding(x).unsqueeze(1)
        return [F.relu(conv(embedded)).squeeze(3).detach().cpu() for conv in model.convs]


def forward_from_embedded_for_xai(model: CNN_Sentiment, embedded: torch.Tensor) -> torch.Tensor:
    """Forward pass that starts from an embedding tensor instead of token ids.

    Gradient x Input은 embedding에 대한 gradient가 필요하다. 일반 `forward()`는
    내부에서 embedding을 만들어 버리므로, gradient 분석용으로 embedding 이후부터의
    연산을 별도 함수로 분리한다.

    Dropout은 intentionally 생략한다. XAI에서는 `model.eval()` 상태의 deterministic한
    설명을 원하고, dropout noise가 attribution을 흔들면 해석이 어려워진다.
    """
    x = embedded.unsqueeze(1)
    pooled = []
    for conv in model.convs:
        conved = F.relu(conv(x)).squeeze(3)
        pool = torch.max(conved, dim=2)[0]
        pooled.append(pool)
    cat = torch.cat(pooled, dim=1)
    return model.fc(cat)
