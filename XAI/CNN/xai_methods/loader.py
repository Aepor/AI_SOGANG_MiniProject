"""CNN checkpoint and vocabulary-cache loading.

노트북은 `best_cnn_model.pt`만 저장하고 vocabulary를 별도 파일로 저장하지 않았다.
그래서 XAI 스크립트는 학습 때와 같은 train split/tokenization으로 vocabulary를 다시
만들고, 그 결과를 `cnn_preprocess_cache.pkl`에 저장해 재사용한다.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from XAI.CNN.xai_methods.model import CNN_Sentiment, infer_architecture_from_state_dict
from XAI.CNN.shared.nsmc_data import clean_nsmc_frame, load_nsmc_raw, split_train_validation
from XAI.CNN.shared.tokenization import make_okt, tokenize_text


# 캐시 구조가 바뀌면 숫자를 올려 오래된 pickle을 무시하게 만든다.
CACHE_VERSION = 1


def build_or_load_vocab_cache(cache_path: Path, refresh_cache: bool = False) -> dict[str, Any]:
    """Load the vocabulary cache or rebuild it from NSMC train data.

    checkpoint의 `embedding.weight` 첫 번째 차원은 vocabulary size와 같아야 한다.
    현재 모델은 25,954개 vocab으로 학습되어 있으므로, train split/stopword/min count가
    조금만 달라져도 checkpoint load가 실패한다.
    """
    if cache_path.exists() and not refresh_cache:
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        if cache.get("version") == CACHE_VERSION:
            return cache

    # 아래 흐름은 nsmc_cnn.ipynb의 전처리 순서를 그대로 따른다.
    train_data, _test_data, train_path, test_path = load_nsmc_raw()
    train_data = clean_nsmc_frame(train_data)
    train_data, _val_data = split_train_validation(train_data, validation_ratio=0.1, seed=42)

    okt = make_okt()
    all_words: list[str] = []
    for text in train_data["document"]:
        all_words.extend(tokenize_text(text, okt))

    # 노트북과 동일하게 2회 이상 등장한 token만 vocabulary에 넣는다.
    # 1회 등장 token은 <unk>로 묶어 모델이 과하게 희소한 단어에 맞춰지지 않게 했다.
    word_counts = Counter(all_words)
    vocab = ["<pad>", "<unk>"] + [word for word, count in word_counts.items() if count >= 2]
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    cache = {
        "version": CACHE_VERSION,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows_after_split": len(train_data),
        "vocab": vocab,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    return cache


def load_cnn_model(
    model_path: Path, vocab_size: int, device: torch.device
) -> tuple[CNN_Sentiment, dict[str, Any]]:
    """Load the CNN checkpoint and verify it matches the reconstructed vocab."""
    state_dict = torch.load(model_path, map_location="cpu")
    arch = infer_architecture_from_state_dict(state_dict)

    # 이 검사는 매우 중요하다. vocab index가 하나라도 밀리면 checkpoint는 로드될 수 있어도
    # 단어와 embedding 의미가 완전히 어긋난다.
    if arch["vocab_size"] != vocab_size:
        raise RuntimeError(
            "Vocab size mismatch. "
            f"Cache vocab={vocab_size}, model embedding={arch['vocab_size']}. "
            "Delete the cache or rerun with --refresh-cache."
        )
    model = CNN_Sentiment(**arch)
    model.load_state_dict(state_dict)
    model.to(device)

    # XAI는 재현 가능한 설명이 중요하므로 dropout을 끈 eval mode로 고정한다.
    model.eval()
    return model, arch
