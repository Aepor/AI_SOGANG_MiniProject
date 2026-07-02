"""NSMC loading and sampling helpers.

이 파일은 데이터프레임 수준의 전처리만 담당한다. 형태소 분석이나 token id 변환은
`tokenization.py`에서 처리한다. 이렇게 나누면 FNN/CNN/Transformer가 같은 NSMC
샘플을 쓰더라도 각 모델의 tokenizer는 따로 유지할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from XAI.CNN.shared.paths import find_data_file


def load_nsmc_raw() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load raw NSMC train/test TSV files and return their resolved paths too."""
    train_path = find_data_file("ratings_train.txt")
    test_path = find_data_file("ratings_test.txt")

    # NSMC 파일은 tab-separated format이고 columns는 id/document/label이다.
    train_data = pd.read_csv(train_path, sep="\t")
    test_data = pd.read_csv(test_path, sep="\t")
    return train_data, test_data, train_path, test_path


def clean_nsmc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same duplicate/null cleanup used in the CNN notebook."""
    frame = frame.copy()

    # 같은 document가 중복되면 train/test 통계와 sample selection이 흐려질 수 있다.
    frame.drop_duplicates(subset=["document"], inplace=True)

    # document가 비어 있으면 tokenizer와 모델 입력 생성이 불가능하므로 제거한다.
    frame.dropna(how="any", inplace=True)
    return frame


def split_train_validation(
    train_data: pd.DataFrame, validation_ratio: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the notebook's train/validation split.

    validation set을 train에서 `sample(frac=0.1, random_state=42)`로 뽑았으므로,
    vocabulary cache도 같은 split을 사용해야 checkpoint의 embedding 크기와 맞는다.
    """
    validation_data = train_data.sample(frac=validation_ratio, random_state=seed)
    train_split = train_data.drop(validation_data.index)
    return train_split, validation_data


def sample_test_pool(test_data: pd.DataFrame, pool_size: int, seed: int) -> pd.DataFrame:
    """Select a reproducible subset of test rows for XAI analysis."""
    pool_size = min(pool_size, len(test_data))
    return test_data.sample(n=pool_size, random_state=seed)
