# XAI 기법 정리: FNN / CNN / Transformer 기반 텍스트 감정분석

이 문서는 **NSMC(Naver Sentiment Movie Corpus) 감정분석 프로젝트**를 기준으로, FNN, CNN, Transformer 모델에서 사용할 수 있는 XAI(Explainable AI) 기법을 정리한 것이다.

프로젝트의 핵심 질문은 다음과 같다.

> 모델은 왜 어떤 리뷰를 긍정 또는 부정으로 판단했는가?

즉, 단순히 성능만 비교하는 것이 아니라, 각 모델이 **어떤 단어, 표현, 문맥, 내부 표현에 의존했는지**를 분석하는 것이 목적이다.

---

## 0. XAI가 설명하는 것의 종류

XAI라고 해서 모두 같은 것을 설명하는 것은 아니다. 텍스트 감정분석에서 XAI는 크게 네 가지 관점으로 나눌 수 있다.

| 종류 | 핵심 질문 | 예시 |
|---|---|---|
| Feature Attribution | 어떤 단어가 예측에 중요했는가? | `지루했다`, `최고`, `ㅋㅋ`, `노잼` |
| Perturbation Analysis | 단어를 지우거나 가리면 예측이 바뀌는가? | `지루했다` 제거 후 부정 확률 감소 |
| Internal Analysis | 모델 내부 filter/head/layer가 무엇을 보고 있는가? | CNN filter가 `너무 지루`에 강하게 반응 |
| Representation Analysis | hidden state 안에 감정 정보가 존재하는가? | 긍정 CLS 평균 - 부정 CLS 평균 |

텍스트 감정분석에서 설명 대상은 이미지의 픽셀이 아니라 **토큰 또는 단어 조각(subword)**이다.

예를 들어 문장이 다음과 같다고 하자.

```text
배우는 좋았지만 스토리는 너무 지루했다.
```

XAI 결과는 이런 식으로 해석될 수 있다.

```text
좋았지만     → 긍정 방향 정보
지루했다     → 부정 방향 정보
너무         → 부정 감정 강화
스토리       → 문맥상 부정 판단에 기여
```

---

# 1. 공통 XAI 기법

먼저 FNN, CNN, Transformer 모두에 적용할 수 있는 공통 기법을 정리한다.

---

## 1.1 Occlusion / Deletion

### 개념

Occlusion은 가장 직관적인 XAI 방법이다.

> 특정 단어를 가리거나 제거했을 때 모델 예측이 얼마나 바뀌는지 보는 방법이다.

예를 들어 원문이 다음과 같다고 하자.

```text
스토리가 너무 지루했다.
```

모델의 원래 예측이 다음과 같다고 하자.

```text
부정 확률: 0.95
```

이제 `지루했다`를 제거하거나 가린다.

```text
스토리가 너무 [MASK].
```

또는

```text
스토리가 너무.
```

그 결과 부정 확률이 다음처럼 변했다고 하자.

```text
부정 확률: 0.95 → 0.52
```

그러면 다음과 같이 해석할 수 있다.

> `지루했다`는 모델이 부정으로 판단하는 데 매우 중요한 표현이다.

### 장점

- 가장 이해하기 쉽다.
- FNN, CNN, Transformer 모두에 적용할 수 있다.
- 발표나 보고서에서 설명하기 좋다.
- gradient 계산이 필요 없다.
- 모델 내부 구조를 몰라도 사용할 수 있다.

### 단점

- 단어를 제거하면 문장이 부자연스러워질 수 있다.
- 모든 토큰마다 다시 예측해야 하므로 계산량이 많다.
- tokenizer가 subword 단위이면 단어 단위 해석과 어긋날 수 있다.
- `[MASK]`로 가리는 방식과 실제 삭제 방식의 결과가 다를 수 있다.

### 프로젝트에서의 사용

이 기법은 반드시 넣는 것이 좋다. 왜냐하면 모델 구조와 상관없이 동일한 기준으로 비교할 수 있기 때문이다.

```text
FNN Occlusion Top-k 단어
CNN Occlusion Top-k 단어
Transformer Occlusion Top-k 단어
```

이렇게 비교할 수 있다.

---

## 1.2 Gradient Saliency

### 개념

Gradient Saliency는 입력이 출력에 얼마나 민감하게 영향을 주는지 보는 방법이다.

> 입력 embedding을 조금 바꿨을 때 출력 점수가 얼마나 변하는지 계산한다.

텍스트 모델에서는 단어 자체가 숫자가 아니므로, 보통 단어 ID가 아니라 **embedding vector**에 대해 gradient를 구한다.

핵심 아이디어는 다음과 같다.

```text
중요도 = ∂ output_score / ∂ input_embedding
```

즉,

```text
출력 점수가 입력 embedding에 대해 얼마나 민감한가?
```

를 보는 것이다.

예를 들어 `지루했다`의 embedding을 조금 바꿨을 때 부정 logit이 크게 변한다면, `지루했다`는 부정 예측에 중요한 토큰이라고 볼 수 있다.

### 장점

- 빠르다.
- 한 번의 backward 연산으로 계산할 수 있다.
- FNN, CNN, Transformer 모두에 적용할 수 있다.

### 단점

- gradient가 noisy할 수 있다.
- gradient saturation 문제가 생길 수 있다.
- “민감도”를 보는 것이므로 실제 원인이라고 단정하면 안 된다.
- 결과가 불안정할 수 있다.

---

## 1.3 Gradient × Input

### 개념

Gradient × Input은 Gradient Saliency보다 조금 더 실용적인 방법이다.

단순히 gradient만 보는 것이 아니라, 실제 입력 embedding 값과 gradient를 곱한다.

```text
중요도 = gradient × input_embedding
```

텍스트에서는 보통 다음처럼 각 토큰별 점수를 계산한다.

```python
score_i = (grad_i * embedding_i).sum()
```

여기서:

```text
grad_i      = i번째 토큰 embedding에 대한 gradient
embedding_i = i번째 토큰의 실제 embedding
score_i     = i번째 토큰의 기여도
```

### 직관

Gradient만 보면 다음을 보는 것이다.

```text
이 방향으로 입력을 바꾸면 출력이 얼마나 변하는가?
```

Gradient × Input은 다음에 더 가깝다.

```text
현재 이 입력이 실제로 출력에 얼마나 기여했는가?
```

### 방향성과 절댓값

Gradient × Input 점수에 `abs()`를 적용하면 중요도의 크기만 보게 된다.

```text
abs 사용      → 중요한 정도만 확인
abs 미사용    → 긍정/부정 방향성까지 확인
```

예를 들어:

```text
지루했다 → 부정 방향 기여
최고     → 긍정 방향 기여
```

인데 `abs()`를 사용하면 둘 다 “중요한 단어”로만 보인다. 따라서 분석 목적에 따라 다르게 사용해야 한다.

### 프로젝트에서의 의미

기존 코드에 있던 `gradient_x_input` 방식은 이 계열에 해당한다. 주의할 점은 이것을 **Integrated Gradients**라고 부르면 안 된다는 것이다.

정확한 이름은:

```text
Gradient × Input
```

이다.

---

## 1.4 Integrated Gradients

### 개념

Integrated Gradients, 줄여서 IG는 gradient 기반 XAI 중에서 많이 사용되는 방법이다.

단순히 현재 입력 한 지점에서 gradient를 보는 것이 아니라, baseline 입력에서 실제 입력까지 이동하면서 gradient를 누적한다.

핵심 흐름은 다음과 같다.

```text
baseline input
↓
조금 실제 입력에 가까운 입력
↓
더 실제 입력에 가까운 입력
↓
actual input
```

이 경로 전체에서 gradient를 계산하고 누적한다.

### 텍스트에서 baseline이란?

텍스트 모델에서 baseline은 “정보가 거의 없는 입력”으로 잡는다.

예시는 다음과 같다.

```text
[PAD] 토큰들
zero embedding
[MASK] 토큰들
```

예를 들어 실제 문장이 다음과 같다고 하자.

```text
정말 최고의 영화였다.
```

baseline은 다음처럼 둘 수 있다.

```text
[PAD] [PAD] [PAD] [PAD]
```

그리고 baseline에서 실제 입력으로 이동하면서, 각 토큰이 긍정 logit 또는 부정 logit에 얼마나 기여했는지 계산한다.

### 예시 결과

```text
정말       +0.21
최고       +1.43
영화였다    +0.12
```

이 경우 `최고`가 긍정 판단에 가장 크게 기여했다고 볼 수 있다.

### 장점

- 단순 Gradient Saliency보다 안정적인 설명을 줄 수 있다.
- 이론적으로 더 정교하다.
- FNN, CNN, Transformer 모두에 적용 가능하다.
- Captum의 `LayerIntegratedGradients`를 사용하면 Transformer embedding layer에도 적용할 수 있다.

### 단점

- 여러 step을 계산해야 하므로 느리다.
- baseline 선택에 따라 결과가 달라질 수 있다.
- Transformer에서는 구현이 Gradient × Input보다 복잡하다.

### 프로젝트에서의 위치

보고서에서는 다음과 같이 설명할 수 있다.

```text
Occlusion은 입력 제거 기반 설명이고,
Integrated Gradients는 gradient 누적 기반 설명이다.
두 결과가 유사하게 나타나는지 비교하였다.
```

---

## 1.5 LIME

### 개념

LIME은 모델 내부 구조를 몰라도 사용할 수 있는 model-agnostic XAI 방법이다.

핵심 아이디어는 다음과 같다.

> 원래 복잡한 모델 주변에서만 작동하는 단순한 설명용 모델을 새로 만든다.

즉, 원래 모델 전체를 해석하는 것이 아니라, **특정 입력 하나 주변에서 모델이 어떻게 행동하는지**를 단순 모델로 근사한다.

### 예시

원문:

```text
배우는 좋았지만 스토리는 너무 지루했다.
```

이 문장에서 단어를 랜덤하게 제거한 여러 문장을 만든다.

```text
배우는 좋았지만 스토리는 너무 [삭제]
배우는 [삭제] 스토리는 너무 지루했다
[삭제] 좋았지만 스토리는 너무 지루했다
배우는 좋았지만 [삭제] 너무 지루했다
```

이렇게 변형된 문장들을 원래 모델에 넣고 예측값을 얻는다. 그다음 이 주변 데이터에 대해 단순한 선형 모델을 학습한다.

이 선형 모델의 가중치를 보면 다음과 같은 해석을 할 수 있다.

```text
지루했다 → 부정 예측에 큰 영향
좋았지만 → 긍정 예측에 영향
```

### 장점

- FNN, CNN, Transformer 모두에 적용 가능하다.
- 모델 내부를 몰라도 된다.
- 설명이 직관적이다.
- 발표용으로 좋다.

### 단점

- 랜덤 perturbation 기반이라 실행할 때마다 결과가 조금 달라질 수 있다.
- 문장을 많이 만들어 예측해야 하므로 느리다.
- local surrogate model이 실제 모델 내부 reasoning과 같다고 단정할 수 없다.
- subword tokenizer와 단어 단위 설명이 어긋날 수 있다.

### 프로젝트에서의 추천 정도

시간이 충분하면 넣어도 좋지만, 필수는 아니다.

추천 우선순위는 다음과 같다.

```text
필수: Occlusion
필수 또는 강력 추천: Integrated Gradients / Gradient × Input
선택: LIME
```

---

## 1.6 SHAP

### 개념

SHAP은 게임이론의 Shapley value 아이디어를 이용해 feature importance를 계산하는 방법이다.

핵심 질문은 다음과 같다.

> 어떤 단어가 여러 단어 조합 속에서 평균적으로 얼마나 기여했는가?

예를 들어 문장이 다음과 같다고 하자.

```text
정말 최고의 영화였다.
```

SHAP은 `최고의`라는 단어가 있을 때와 없을 때, 다양한 단어 조합에 걸쳐 긍정 확률이 얼마나 변하는지 계산한다.

### 장점

- 이론적으로 설득력이 강하다.
- feature contribution 관점이 명확하다.
- 모델 비교 설명에 좋다.

### 단점

- 정확히 계산하려면 조합 수가 매우 많다.
- Transformer 텍스트 모델에서는 매우 느릴 수 있다.
- 구현과 계산 비용이 크다.
- 프로젝트 규모가 커질 수 있다.

### 프로젝트에서의 추천 정도

직접 구현 필수는 아니다. 보고서의 관련 기법 소개 또는 한계/확장 방향으로 언급하기 좋다.

---

# 2. FNN에서의 XAI

FNN은 일반적으로 다음과 같은 구조를 가진다.

```text
문장
↓
tokenizer
↓
embedding
↓
평균 pooling 또는 flatten
↓
fully connected layers
↓
positive / negative
```

FNN은 구조가 단순하기 때문에 설명도 상대적으로 쉽다.

---

## 2.1 Weight-based Explanation

입력이 BoW, TF-IDF, one-hot 같은 형태라면 weight를 직접 해석할 수 있다.

예를 들어 단어가 직접 feature라면 다음처럼 볼 수 있다.

```text
최고      → positive weight 큼
지루      → negative weight 큼
노잼      → negative weight 큼
감동      → positive weight 큼
```

하지만 프로젝트에서 FNN도 tokenizer + embedding을 사용한다면, 단어가 직접 feature가 아니라 embedding으로 바뀌기 때문에 단순 weight 해석은 어렵다.

따라서 embedding 기반 FNN에서는 gradient 기반 XAI가 더 자연스럽다.

---

## 2.2 FNN + Gradient / Gradient × Input

FNN에서 가장 자연스러운 XAI는 embedding에 대한 gradient를 보는 것이다.

흐름은 다음과 같다.

```text
각 토큰 embedding
↓
FNN
↓
positive logit 또는 negative logit
↓
gradient 계산
↓
토큰별 중요도 계산
```

예를 들어 문장이 다음과 같다고 하자.

```text
스토리가 너무 지루했다.
```

target을 negative logit으로 잡았을 때 결과가 다음과 같을 수 있다.

```text
스토리가   0.12
너무       0.34
지루했다   1.28
```

이 경우 `지루했다`가 부정 예측에 가장 크게 기여했다고 해석한다.

---

## 2.3 FNN + Integrated Gradients

FNN에서도 IG를 사용할 수 있다.

baseline을 `[PAD]` 문장으로 두고 실제 문장까지 interpolation하면서 attribution을 계산한다.

```text
baseline: [PAD] [PAD] [PAD] [PAD]
input:    스토리가 너무 지루했다
```

결과를 토큰별로 합산하면 각 단어의 중요도를 얻을 수 있다.

FNN은 Transformer보다 구조가 단순하기 때문에 IG 구현이 상대적으로 쉽다.

---

## 2.4 FNN + Occlusion

FNN에서도 단어를 하나씩 제거하거나 가려서 확률 변화를 볼 수 있다.

예시:

```text
원문: 배우는 좋았지만 스토리는 지루했다
부정 확률: 0.91

"지루했다" 제거
부정 확률: 0.61

"좋았지만" 제거
부정 확률: 0.96
```

해석:

```text
지루했다는 부정 판단의 핵심 근거이다.
좋았지만은 긍정 방향 정보를 제공한다.
```

---

## 2.5 FNN 분석의 핵심 포인트

FNN 분석에서는 다음 질문을 던질 수 있다.

```text
FNN은 어떤 단어 embedding에 가장 크게 반응하는가?
FNN은 단어 순서 없이도 감정 단어를 잘 잡는가?
FNN은 "좋았지만 지루했다" 같은 대조 구조를 잘 처리하는가?
```

---

## 2.6 보고서 문장 예시

```text
FNN은 문장 구조를 깊게 반영하기보다 입력 토큰 embedding을 feed-forward layer에 통과시켜 분류한다. 따라서 본 프로젝트에서는 gradient 기반 attribution과 occlusion을 통해 각 토큰이 출력 logit에 미치는 영향을 분석하였다.
```

---

# 3. CNN에서의 XAI

텍스트 CNN은 이미지 CNN과 비슷하게 convolution filter를 사용하지만, 이미지의 공간 패턴이 아니라 **연속된 단어 패턴**, 즉 n-gram을 잡아낸다.

구조는 대략 다음과 같다.

```text
문장 토큰
↓
embedding sequence
↓
1D convolution filters
↓
ReLU
↓
max pooling
↓
linear classifier
↓
positive / negative
```

---

## 3.1 Filter Activation Analysis

CNN에서 가장 중요한 XAI는 filter activation 분석이다.

> 각 filter가 어떤 n-gram에 강하게 반응했는지 보는 방법이다.

예를 들어 kernel size가 3인 filter는 연속된 3개 토큰을 본다.

문장:

```text
정말 너무 재미있었다
```

3-gram 후보는 다음과 같다.

```text
정말 너무 재미
너무 재미 있었다
```

어떤 filter가 `너무 재미 있었다`에서 강하게 활성화되면, 이 filter는 긍정 표현 패턴에 반응한다고 해석할 수 있다.

### 예시 1: 긍정 filter

Filter 12가 강하게 반응한 n-gram들이 다음과 같다고 하자.

```text
너무 재밌
정말 최고
완전 감동
```

해석:

```text
Filter 12는 긍정 감정 표현을 탐지하는 필터로 볼 수 있다.
```

### 예시 2: 부정 filter

Filter 27이 강하게 반응한 n-gram들이 다음과 같다고 하자.

```text
시간 아깝
너무 지루
최악 영화
```

해석:

```text
Filter 27은 부정 감정 표현을 탐지하는 필터로 볼 수 있다.
```

---

## 3.2 Max-pooling 위치 분석

텍스트 CNN에서는 convolution 후 max pooling을 자주 사용한다.

즉, 각 filter가 문장 전체를 훑은 뒤 가장 강하게 반응한 위치 하나를 선택한다.

```text
가장 강하게 반응한 위치 = 모델이 중요하게 본 n-gram 후보
```

예를 들어 문장이 다음과 같다고 하자.

```text
배우는 좋았지만 스토리는 너무 지루했다
```

어떤 filter의 max activation 위치가 다음 구간이었다고 하자.

```text
너무 지루했다
```

해석:

```text
이 CNN filter는 이 문장에서 "너무 지루했다" 부분을 핵심 부정 패턴으로 잡았다.
```

---

## 3.3 CNN + Gradient / Integrated Gradients

CNN도 embedding input에 대해 gradient를 구할 수 있다.

흐름은 다음과 같다.

```text
토큰 embedding
↓
Conv1D
↓
Pooling
↓
Classifier
↓
target logit에 대한 gradient
```

결과는 token-level attribution으로 나온다.

CNN에서는 attribution이 단일 단어보다 주변 단어 묶음에 퍼질 수 있다. 왜냐하면 convolution filter가 연속된 토큰들을 함께 보기 때문이다.

예를 들어:

```text
너무 지루했다
```

전체가 하나의 부정 n-gram처럼 작동하면, `너무`, `지루`, `했다`에 함께 높은 점수가 생길 수 있다.

---

## 3.4 CNN + Occlusion / n-gram Occlusion

CNN에서는 단어 하나씩 제거하는 것보다 n-gram 단위 제거가 더 의미 있을 수 있다.

이유는 CNN filter가 kernel size 2, 3, 4처럼 연속된 구간을 보기 때문이다.

예를 들어 문장이 다음과 같다고 하자.

```text
이 영화는 정말 시간 아까웠다
```

단어 하나씩 제거하는 것보다:

```text
시간 제거
아까웠다 제거
```

다음처럼 구 단위로 제거하는 것이 더 자연스럽다.

```text
시간 아까웠다 제거
```

### 추천 방식

CNN에서는 다음 세 가지 occlusion을 비교할 수 있다.

```text
unigram occlusion: 단어 1개씩 제거
bigram occlusion: 연속 2개 토큰 제거
trigram occlusion: 연속 3개 토큰 제거
```

CNN kernel size와 맞춰서 분석하면 좋다.

---

## 3.5 Grad-CAM for Text CNN

Grad-CAM은 원래 이미지 CNN에서 어느 이미지 영역이 class 판단에 중요했는지 보여주는 기법이다.

텍스트 CNN에 적용하면 개념적으로는 다음을 보는 방식이 된다.

```text
어느 토큰 위치 또는 n-gram 위치가 class 판단에 중요했는가?
```

다만 텍스트 CNN에서는 일반적으로 Grad-CAM보다 다음 기법들이 더 직관적이다.

```text
filter activation
max-pooling 위치
n-gram occlusion
gradient attribution
```

따라서 프로젝트에서는 Grad-CAM을 필수로 넣기보다는, 관련 기법 또는 확장 방향 정도로 언급해도 충분하다.

---

## 3.6 CNN 분석의 핵심 포인트

CNN 분석에서는 다음 질문을 던질 수 있다.

```text
CNN filter가 어떤 감정 n-gram을 잡는가?
"너무 지루", "완전 최고", "시간 아깝" 같은 구간에 반응하는가?
FNN보다 구 표현을 더 잘 잡는가?
```

---

## 3.7 보고서 문장 예시

```text
CNN은 convolution filter를 통해 연속된 토큰 패턴을 탐지하므로, 단일 단어뿐 아니라 n-gram 단위의 감정 표현을 분석할 수 있다. 본 프로젝트에서는 filter activation, max-pooling 위치, n-gram occlusion을 통해 CNN이 어떤 표현 패턴에 반응하는지 확인하였다.
```

---

# 4. Transformer에서의 XAI

Transformer는 FNN, CNN보다 훨씬 복잡하다.

KcELECTRA 같은 모델은 대략 다음과 같은 구조를 가진다.

```text
token embedding
↓
Transformer layer 1
↓
Transformer layer 2
↓
...
↓
Transformer layer 12
↓
[CLS] hidden state
↓
classification head
↓
positive / negative
```

Transformer XAI에서는 크게 네 가지를 볼 수 있다.

```text
1. 입력 토큰 중요도
2. Attention pattern
3. Hidden state / CLS representation
4. Head/layer ablation
```

---

## 4.1 Transformer + Occlusion

Transformer에서도 가장 직관적인 방법은 occlusion이다.

문장:

```text
배우는 좋았지만 스토리는 너무 지루했다
```

원래 예측:

```text
부정 확률: 0.93
```

각 토큰을 하나씩 `[MASK]` 처리하거나 제거한다.

```text
[MASK]는 좋았지만 스토리는 너무 지루했다
배우는 [MASK] 스토리는 너무 지루했다
배우는 좋았지만 [MASK]는 너무 지루했다
...
```

각 경우의 부정 확률 변화를 본다.

예를 들어:

```text
지루했다 제거 → 부정 확률 크게 감소
좋았지만 제거 → 부정 확률 증가
```

해석:

```text
지루했다는 부정 판단 근거이다.
좋았지만은 긍정 방향 정보를 제공한다.
```

### 주의점

Transformer에서 `[MASK]`를 사용하는 방식은 직관적이지만, 실제 fine-tuning 시 입력 분포와 다를 수 있다.

따라서 다음 중 하나를 정해서 일관되게 사용해야 한다.

```text
[MASK]로 대체
[PAD]로 대체
토큰 삭제
```

결과 해석에서는 이 선택을 명시하는 것이 좋다.

---

## 4.2 Transformer + Integrated Gradients

Transformer에서 가장 추천할 만한 attribution 방법 중 하나가 Integrated Gradients이다.

Token id 자체는 discrete하기 때문에 직접 gradient를 주기 어렵다. 따라서 보통 embedding layer에 대해 attribution을 계산한다.

흐름은 다음과 같다.

```text
input_ids
↓
embedding layer
↓
Transformer
↓
classification head
↓
target logit
```

여기서 embedding layer에 대해 IG를 적용한다.

### 결과 예시

문장:

```text
배우는 좋았지만 스토리는 너무 지루했다
```

positive logit 기준 attribution이 다음과 같다고 하자.

```text
배우는       +0.12
좋았지만     +0.48
스토리는     -0.10
너무         -0.33
지루했다     -1.24
```

이 경우 해석은 다음과 같다.

```text
좋았지만은 긍정 판단에 기여했다.
지루했다는 긍정 판단을 약화하고, 반대로 부정 판단에 기여했다.
```

### target class 주의

Attribution을 계산할 때 target을 무엇으로 잡는지가 중요하다.

가능한 기준:

```text
positive logit 기준
negative logit 기준
모델이 예측한 class 기준
정답 label 기준
```

보고서에서는 반드시 어떤 target 기준으로 attribution을 계산했는지 밝혀야 한다.

추천은 다음 중 하나이다.

```text
모델이 예측한 class 기준 attribution
정답 class 기준 attribution
```

둘 다 분석하면 더 좋지만, 하나만 선택해도 된다.

---

## 4.3 Attention Visualization

Transformer의 대표적인 내부 구조는 attention이다.

Attention은 대략 다음을 의미한다.

```text
각 토큰이 다른 토큰을 얼마나 참고했는가?
```

예를 들어 마지막 layer에서 `[CLS]`가 `지루했다`에 높은 attention을 준다면, 문장 분류와 관련된 패턴일 수 있다.

### 예시

문장:

```text
배우는 좋았지만 스토리는 너무 지루했다
```

마지막 layer `[CLS]` attention이 다음과 같다고 하자.

```text
좋았지만     0.12
스토리는     0.08
너무         0.18
지루했다     0.41
```

해석:

```text
마지막 layer에서 [CLS]는 "지루했다"에 가장 크게 attention을 두었다.
```

### 중요한 주의점

Attention을 바로 “모델의 설명”이라고 단정하면 위험하다.

Attention weight는 정보 흐름의 한 단면을 보여줄 뿐이며, 실제 예측에 대한 인과적 기여를 완전히 보장하지 않는다.

따라서 보고서에서는 다음처럼 쓰는 것이 안전하다.

```text
Attention map은 모델 내부에서 토큰 간 정보 흐름을 보여주는 보조적 분석으로 사용하였다. 단, attention weight만으로 모델의 최종 판단 근거를 완전히 설명한다고 보지는 않았다.
```

---

## 4.4 Attention Rollout

Transformer는 여러 layer가 쌓여 있기 때문에 마지막 layer attention만 보면 부족할 수 있다.

Attention Rollout은 여러 layer의 attention을 누적해서 입력 토큰이 최종 CLS 표현에 얼마나 영향을 미쳤는지 근사적으로 보는 방법이다.

개념적으로는 다음과 같다.

```text
Layer 1 attention
×
Layer 2 attention
×
...
×
Layer 12 attention
```

즉, 여러 layer를 지나며 attention이 어떻게 전달되는지 누적해서 본다.

### 장점

- 마지막 layer attention보다 전체 attention 흐름을 더 잘 반영하려고 한다.
- Transformer 구조 분석에 적합하다.

### 단점

- 여전히 attention 기반이므로 실제 원인이라고 단정할 수 없다.
- 구현이 단순 attention visualization보다 복잡하다.

---

## 4.5 Attention × Gradient

Attention × Gradient는 attention weight만 보는 것이 아니라, attention에 대한 gradient까지 함께 보는 방법이다.

핵심 질문은 다음과 같다.

```text
attention 값이 클 뿐 아니라, 그 attention이 출력에 민감하게 영향을 주는가?
```

예를 들어 어떤 토큰에 attention이 높더라도, 그 attention이 최종 output에 큰 영향을 주지 않을 수 있다.

Attention × Gradient는 이 문제를 줄이기 위해 사용된다.

### 장점

- 단순 attention보다 class-specific한 설명을 만들 수 있다.
- attention weight와 출력 민감도를 함께 고려한다.

### 단점

- 구현이 더 어렵다.
- gradient 기반 방법이므로 noisy할 수 있다.
- 프로젝트 필수 기법은 아니다.

---

## 4.6 CLS Hidden State 분석

Transformer 분류 모델에서는 보통 마지막 layer의 `[CLS]` hidden state가 classification head로 들어간다.

```text
마지막 layer CLS
↓
linear classifier
↓
positive / negative
```

따라서 `[CLS]`는 문장 전체 표현으로 자주 사용된다.

너희 프로젝트에서 나중에 할 Sentiment Vector도 이 CLS를 이용한다.

```text
v_sentiment = mean(CLS_positive) - mean(CLS_negative)
```

이 벡터를 만들고, 각 문장의 CLS를 이 방향에 projection하면 다음을 볼 수 있다.

```text
이 방향으로 갈수록 긍정인가?
반대 방향으로 갈수록 부정인가?
```

---

## 4.7 Layer별 CLS 분석

마지막 layer만 보는 것이 아니라 layer 0~12의 CLS를 모두 비교할 수도 있다.

예상되는 분석 예시는 다음과 같다.

```text
Layer 0 CLS: 감정 구분 약함
Layer 6 CLS: 감정 정보 증가
Layer 12 CLS: 감정 구분 강함
```

이런 결과가 나오면 다음과 같이 해석할 수 있다.

```text
fine-tuning된 Transformer 내부에서 감정 정보가 layer를 거치며 점점 분리된다.
```

단, 실제 결과는 반드시 실험으로 확인해야 한다. 마지막 layer보다 중간 layer가 더 좋은 경우도 있을 수 있다.

---

## 4.8 Linear Probe

Linear Probe는 hidden state에 특정 정보가 들어 있는지 확인하는 방법이다.

예를 들어 KcELECTRA의 CLS hidden state를 고정하고, 그 위에 아주 단순한 logistic regression만 학습한다.

```text
CLS hidden state
↓
linear classifier
↓
positive / negative
```

만약 이 단순 classifier가 잘 맞춘다면, 다음과 같이 해석할 수 있다.

```text
CLS hidden state 안에 이미 감정 정보가 선형적으로 분리되어 있다.
```

Linear Probe는 엄밀히 말하면 XAI라기보다는 representation analysis에 가깝다. 하지만 프로젝트의 심화 분석으로 매우 좋다.

---

## 4.9 Head Ablation

Transformer에는 여러 attention head가 있다.

예를 들어 base 모델이 12 layer, 각 layer 12 head라면 총 144개의 attention head가 있다.

Head ablation은 특정 attention head를 꺼보고 성능이나 예측이 얼마나 변하는지 보는 방법이다.

예를 들어:

```text
Layer 3의 Head 7 제거
↓
부정 확률 크게 감소
```

이면 다음과 같이 해석할 수 있다.

```text
해당 head가 감정 판단에 중요할 가능성이 있다.
```

### 장점

- Transformer 내부 구조 분석에 좋다.
- Mechanistic Interpretability 느낌이 난다.

### 단점

- 구현 난이도가 높다.
- 실험 범위가 커진다.
- 단순 XAI 프로젝트에서는 필수는 아니다.

---

## 4.10 Transformer 분석의 핵심 포인트

Transformer 분석에서는 다음 질문을 던질 수 있다.

```text
Transformer는 문맥 반전을 잘 반영하는가?
"좋았지만 지루했다" 같은 문장에서 마지막 부정 표현을 잘 잡는가?
Attention과 IG 결과가 일치하는가?
CLS hidden state에서 감정 방향이 분리되는가?
layer가 깊어질수록 감정 정보가 더 잘 분리되는가?
```

---

## 4.11 보고서 문장 예시

```text
Transformer 기반 모델에서는 입력 토큰별 attribution뿐 아니라 attention pattern과 [CLS] hidden representation을 함께 분석하였다. Attention은 토큰 간 정보 흐름을 보여주는 보조 지표로 사용하고, Integrated Gradients 및 Occlusion을 통해 실제 예측 확률 변화와 feature attribution을 비교하였다.
```

---

# 5. FNN / CNN / Transformer별 추천 XAI 조합

모든 기법을 다 구현하려고 하면 프로젝트가 너무 커진다. 따라서 모델별 핵심 기법을 정하는 것이 좋다.

## 5.1 FNN 추천 기법

```text
1. Occlusion
2. Gradient × Input
3. Integrated Gradients
```

분석 포인트:

```text
FNN은 어떤 단어 embedding에 가장 크게 반응하는가?
단어 순서나 n-gram 없이도 감정 단어를 잘 잡는가?
```

## 5.2 CNN 추천 기법

```text
1. Occlusion
2. n-gram Occlusion
3. Filter Activation
4. Max-pooling 위치 분석
5. Gradient × Input 또는 Integrated Gradients
```

분석 포인트:

```text
CNN filter가 어떤 감정 n-gram을 잡는가?
"너무 지루", "완전 최고", "시간 아깝" 같은 구간에 반응하는가?
FNN보다 구 표현을 더 잘 잡는가?
```

## 5.3 Transformer 추천 기법

```text
1. Occlusion
2. Integrated Gradients
3. Attention Visualization
4. Attention Rollout 또는 Attention × Gradient
5. CLS Sentiment Direction
6. Layer별 CLS 분석
```

분석 포인트:

```text
Transformer는 문맥 반전을 잘 반영하는가?
"좋았지만 지루했다" 같은 문장에서 마지막 부정 표현을 잘 잡는가?
Attention과 IG 결과가 일치하는가?
CLS hidden state에서 감정 방향이 분리되는가?
```

---

# 6. 예시 문장으로 세 모델 비교

예시 문장:

```text
배우는 좋았지만 스토리는 너무 지루했다.
```

## 6.1 FNN 예상 분석

FNN은 단어별 감정에 반응할 가능성이 크다.

```text
좋았지만 → 긍정 방향
지루했다 → 부정 방향
너무     → 부정 강화
```

하지만 문장 구조를 깊게 보지 못하면, `좋았지만`과 `지루했다`가 충돌할 수 있다.

즉 FNN은 다음 질문을 통해 분석할 수 있다.

```text
FNN은 최종 부정 판단에서 "지루했다"를 충분히 중요하게 봤는가?
FNN은 "좋았지만"이라는 긍정 표현 때문에 혼동했는가?
```

## 6.2 CNN 예상 분석

CNN은 구 단위 표현에 반응할 가능성이 크다.

```text
너무 지루했다 → 강한 부정 n-gram
배우는 좋았지만 → 약한 긍정/전환 표현
```

CNN filter가 `너무 지루` 같은 구간에서 크게 활성화되면 좋은 분석 포인트가 된다.

분석 질문:

```text
CNN filter가 "너무 지루했다" 구간에서 max activation을 보였는가?
CNN은 FNN보다 연속 표현을 더 잘 잡았는가?
```

## 6.3 Transformer 예상 분석

Transformer는 문맥과 대조 구조를 더 잘 반영할 가능성이 크다.

```text
좋았지만 → 앞부분 긍정
하지만   → 대조 신호
너무 지루했다 → 최종 부정 판단 근거
```

Attention이나 IG에서 `하지만`, `지루했다`가 중요하게 나오면 다음처럼 해석할 수 있다.

```text
Transformer는 단순 감정 단어뿐 아니라 문장 내 대조 구조를 반영했다.
```

---

# 7. XAI 결과 해석 시 주의점

## 7.1 XAI는 정답이 아니라 근사 설명이다

Occlusion, IG, LIME, SHAP, Attention 모두 모델의 판단 과정을 완벽하게 보여주는 것은 아니다.

각 기법은 서로 다른 가정을 기반으로 한다.

```text
Occlusion → 입력을 제거했을 때 예측이 바뀌는가?
IG → baseline에서 입력까지 이동하며 gradient가 어떻게 누적되는가?
LIME → 입력 주변에서 단순 모델로 근사했을 때 어떤 feature가 중요한가?
SHAP → 다양한 feature 조합에서 평균 기여도가 어떤가?
Attention → 토큰 간 정보 흐름이 어떻게 나타나는가?
```

따라서 보고서에서는 다음처럼 쓰는 것이 좋다.

```text
본 프로젝트의 XAI 결과는 모델 내부 판단의 완전한 증명이 아니라, 입력 토큰과 예측 사이의 관련성을 추정하는 분석 도구로 사용하였다.
```

## 7.2 Subword 문제

KcELECTRA tokenizer는 단어를 subword로 쪼갤 수 있다.

예:

```text
지루했다
→ 지루
→ ##했다
```

또는 tokenizer에 따라 다른 방식으로 나뉠 수 있다.

XAI 점수도 subword별로 나오므로, 보고서나 발표에서는 subword를 다시 합쳐서 보여주는 것이 좋다.

```text
지루 + ##했다 → 지루했다
```

## 7.3 Target class를 명확히 해야 한다

Attribution을 계산할 때 target을 무엇으로 잡는지가 중요하다.

가능한 기준:

```text
positive logit 기준 attribution
negative logit 기준 attribution
predicted class 기준 attribution
true label 기준 attribution
```

추천은 다음 중 하나를 명확히 정하는 것이다.

```text
모델이 예측한 class 기준 attribution
정답 class 기준 attribution
```

보고서에서는 반드시 다음을 밝혀야 한다.

```text
본 분석에서는 모델이 예측한 class logit을 target으로 하여 attribution을 계산하였다.
```

또는

```text
본 분석에서는 정답 label에 해당하는 logit을 target으로 하여 attribution을 계산하였다.
```

## 7.4 Attention은 보조 분석으로 사용해야 한다

Attention map은 시각적으로 좋고 설명하기 쉽지만, attention weight만으로 “모델이 이 단어 때문에 예측했다”고 단정하면 안 된다.

따라서 attention은 다음 기법들과 함께 비교하는 것이 좋다.

```text
Occlusion
Integrated Gradients
Gradient × Input
```

보고서 문장:

```text
Attention map은 토큰 간 정보 흐름을 보여주는 보조적 분석으로 사용하였으며, 최종 판단 근거는 Occlusion 및 Integrated Gradients 결과와 함께 비교하여 해석하였다.
```

---

# 8. 프로젝트 최종 추천 구성

너희 프로젝트가 너무 커지지 않으면서도 충분히 깊이 있어 보이려면 다음 구성이 좋다.

## 8.1 공통 실험

FNN, CNN, Transformer 모두에 대해 다음을 수행한다.

```text
1. Accuracy, Precision, Recall, F1 비교
2. Occlusion 분석
3. Gradient × Input 또는 Integrated Gradients 분석
```

## 8.2 CNN 전용 실험

CNN에서는 다음을 추가한다.

```text
1. Filter activation top n-gram 분석
2. Max-pooling 위치 분석
3. n-gram occlusion
```

## 8.3 Transformer 전용 실험

Transformer에서는 다음을 추가한다.

```text
1. Attention visualization
2. IG와 Attention 비교
3. CLS sentiment direction
4. Layer별 CLS 감정 분리도 분석
```

---

# 9. 최종 보고서 구조 예시

```text
1. 모델별 성능 비교
   - FNN / CNN / Transformer Accuracy, F1

2. 공통 XAI 분석
   - Occlusion
   - Integrated Gradients

3. FNN 분석
   - 단어별 gradient attribution

4. CNN 분석
   - filter activation
   - n-gram pattern
   - max-pooling 위치

5. Transformer 분석
   - attention map
   - IG attribution
   - CLS sentiment vector

6. 모델별 차이 해석
   - FNN: 단어 중심
   - CNN: n-gram 패턴 중심
   - Transformer: 문맥과 대조 구조 중심

7. 한계
   - XAI는 근사 설명
   - attention만으로 설명 불가
   - subword 처리 문제
   - baseline 선택 문제
```

---

# 10. 최종 요약

너희 프로젝트에서는 다음처럼 정리하면 가장 좋다.

```text
FNN: 단어별 attribution 중심
CNN: n-gram/filter activation 중심
Transformer: IG + Attention + CLS sentiment direction 중심
```

공통 비교 축은 다음으로 두는 것이 좋다.

```text
Occlusion + Integrated Gradients
```

이렇게 하면 단순히 “Transformer가 성능이 좋다”가 아니라, 각 모델이 다른 방식으로 감정 정보를 처리한다는 점을 보여줄 수 있다.

최종적으로 보고서에서는 다음과 같이 주장할 수 있다.

```text
FNN은 개별 단어 감정에 주로 반응했고,
CNN은 연속된 감정 표현 패턴에 반응했으며,
Transformer는 문맥과 대조 구조까지 반영하는 경향을 보였다.
```

---

# 11. 발표용 한 문장 요약

```text
본 프로젝트는 FNN, CNN, Transformer 감정분류 모델에 XAI 기법을 적용하여, 각 모델이 단어, n-gram, 문맥 정보를 어떻게 활용하는지 비교 분석한다.
```
