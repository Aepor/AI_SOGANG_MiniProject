"""Tokenization helpers that mirror `XAI/CNN/nsmc_cnn.ipynb`.

이 파일은 CNN XAI 전체에서 가장 중요한 전처리 기준을 한곳에 모아 둔다.
학습 당시 노트북과 tokenization 방식이 달라지면 `best_cnn_model.pt`가 학습한
vocabulary index와 새로 만든 입력 index가 어긋날 수 있으므로, 반드시 이 helper를
통해서만 리뷰 문장을 token/id로 바꾸는 것이 안전하다.
"""

from __future__ import annotations

from konlpy.tag import Okt


# CNN 노트북에서 사용한 불용어 목록을 그대로 복사했다.
# XAI 결과는 이 목록으로 제거된 단어를 설명할 수 없다. 예를 들어 "는" 같은 조사는
# 원문에는 있어도 model input에는 없으므로 occlusion/filter 결과에 등장하지 않는다.
STOPWORDS = [
    "의",
    "가",
    "이",
    "은",
    "들",
    "는",
    "좀",
    "잘",
    "걍",
    "과",
    "도",
    "를",
    "으로",
    "자",
    "에",
    "와",
    "한",
    "하다",
]


def make_okt() -> Okt:
    """Create the KoNLPy Okt tokenizer used by the original notebook.

    KoNLPy는 내부적으로 JPype/JVM을 사용한다. 그래서 Python 패키지가 설치되어 있어도
    JDK 또는 JAVA_HOME 설정이 깨져 있으면 `Okt()` 생성에서 실패할 수 있다.
    이 함수는 그런 환경 문제를 XAI 코드 쪽에서 더 읽기 쉬운 오류로 바꿔 준다.
    """
    try:
        return Okt()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize KoNLPy Okt. Check JPype/JDK/JAVA_HOME in this interpreter."
        ) from exc


def tokenize_text(text: str, okt: Okt | None = None) -> list[str]:
    """Tokenize one review into the exact token unit used by the CNN model.

    Args:
        text: 원문 영화 리뷰.
        okt: 이미 만들어 둔 Okt 객체. 여러 문장을 처리할 때 새로 만들지 않고 재사용하면 빠르다.

    Returns:
        stem=True 형태소 분석 후 stopword를 제거한 token 목록.
    """
    analyzer = okt if okt is not None else make_okt()

    # stem=True는 "봤어요" -> "보다", "지루했다" -> "지루하다"처럼 어근을 맞춘다.
    # 학습 때도 이 옵션을 썼기 때문에 XAI에서도 반드시 같은 값을 써야 한다.
    tokens = analyzer.morphs(str(text), stem=True)
    return [word for word in tokens if word not in STOPWORDS]


def encode_tokens(
    tokens: list[str], word_to_index: dict[str, int], max_len: int
) -> tuple[list[int], int]:
    """Convert tokens to the padded integer sequence expected by TextCNN.

    CNN은 고정 길이 `[max_len]` id sequence를 입력으로 받는다. 실제 token 길이가
    짧으면 `<pad>`를 붙이고, 길면 학습 때와 같이 앞 `max_len`개만 남긴다.

    Returns:
        ids: padding/truncation까지 끝난 길이 `max_len`의 정수 id 목록.
        original_len: padding을 제외한 실제 입력 token 수. XAI 결과에서 `<pad>` 위치를
            제외하기 위해 같이 반환한다.
    """
    pad_idx = word_to_index.get("<pad>", 0)
    unk_idx = word_to_index.get("<unk>", 1)

    # max_len 뒤의 token은 모델에 들어가지 않는다. 따라서 XAI 설명 대상도 아니다.
    trimmed = tokens[:max_len]

    # 학습 vocabulary에 없던 token은 모두 <unk> 하나로 합쳐진다. 이 경우 token의
    # 원문 의미와 embedding 의미가 약해질 수 있으므로 해석 때 주의해야 한다.
    ids = [word_to_index.get(token, unk_idx) for token in trimmed]
    original_len = len(ids)
    if len(ids) < max_len:
        ids = ids + [pad_idx] * (max_len - len(ids))
    return ids, original_len


def decode_ids(ids: list[int], index_to_word: dict[int, str]) -> list[str]:
    """Best-effort inverse mapping from ids back to vocabulary tokens."""
    return [index_to_word.get(int(idx), "<unk>") for idx in ids]
