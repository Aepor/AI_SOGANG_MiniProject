# NSMC CNN 전용 XAI 분석 계획

## 1. 목적과 범위

이 문서는 `XAI/CNN/XAI_기법_정리_FNN_CNN_Transformer.md`의 CNN 파트를 바탕으로, `XAI/CNN/nsmc_cnn.ipynb`에 구현된 CNN 감정분석 코드에 맞춘 XAI 분석 계획을 정리한다.

분석 대상은 CNN 모델 하나로 제한한다. FNN, Transformer, Attention, CLS hidden state, Attention Rollout 같은 Transformer 전용 분석은 이 계획에서 제외한다.

최종 목표는 다음 질문에 답하는 것이다.

```text
1. 현재 CNN 모델은 어떤 형태소 또는 n-gram을 근거로 긍정/부정을 판단하는가?
2. filter size 3, 4, 5의 convolution filter는 어떤 감정 표현 패턴에 강하게 반응하는가?
3. max-pooling이 선택한 위치는 실제 감정 핵심 구간과 일치하는가?
4. 단어 단위 occlusion과 n-gram occlusion 중 CNN 판단 근거를 더 잘 설명하는 방식은 무엇인가?
5. 오분류 사례에서 CNN이 잘못 반응한 n-gram은 무엇인가?
```

## 2. 현재 CNN 코드 요약

현재 노트북의 흐름은 다음과 같다.

```text
NSMC 데이터 로드
-> 중복/결측 제거
-> train/validation/test 분리
-> Okt 형태소 분석과 stopword 제거
-> train 데이터 기준 vocabulary 생성
-> token id sequence 생성, max_len=30으로 padding/truncation
-> CNN_Sentiment 학습
-> best_cnn_model.pt 저장
-> metrics.json 저장
-> best model test 평가
-> ipywidgets 예측 UI
```

현재 모델 구조는 다음과 같다.

| 항목 | 현재 코드 값 |
|---|---:|
| tokenizer | `Okt().morphs(text, stem=True)` |
| stopwords | `['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']` |
| max length | `30` |
| embedding dim | `128` |
| convolution | `nn.Conv2d(1, 100, kernel_size=(fs, 128))` |
| filter sizes | `[3, 4, 5]` |
| filters per size | `100` |
| pooling | sequence dimension max-pooling |
| classifier | `Linear(300, 2)` |
| dropout | `0.5` |
| classes | `0=부정`, `1=긍정` |

현재 `metrics.json` 기준 성능은 다음과 같다.

```text
best validation accuracy: epoch 4, 약 0.8548
test accuracy: 약 0.8495
```

따라서 XAI 분석은 "성능이 충분히 학습된 CNN 모델이 어떤 형태소 n-gram을 근거로 판단했는가"를 설명하는 방향으로 설계한다.

## 3. CNN에 맞는 XAI 우선순위

정리 문서의 CNN 추천 기법 중, 현재 노트북 구조에 가장 잘 맞는 순서는 다음과 같다.

| 우선순위 | 기법 | 사용 목적 | 필수 여부 |
|---|---|---|---|
| 1 | n-gram Occlusion | CNN kernel size와 맞춰 입력 구간 제거 영향 측정 | 필수 |
| 2 | Filter Activation Analysis | 각 conv filter가 강하게 반응한 n-gram 추출 | 필수 |
| 3 | Max-pooling 위치 분석 | 각 filter가 문장에서 최종 선택한 위치 확인 | 필수 |
| 4 | Unigram Occlusion | 단어 하나씩 가렸을 때 예측 변화 비교 | 필수 |
| 5 | Gradient x Input | embedding 기준 token-level attribution 확인 | 권장 |
| 6 | Integrated Gradients | `<pad>` baseline부터 입력 embedding까지 누적 gradient 분석 | 구현 완료 |
| 7 | Grad-CAM for Text CNN | conv feature map 기반 class activation 확인 | 선택, 발표 확장용 |

이 프로젝트에서는 다음 5개를 구현하면 충분하다.

```text
1. Unigram Occlusion
2. n-gram Occlusion
3. Filter Activation Analysis
4. Max-pooling 위치 분석
5. Gradient x Input
```

Integrated Gradients는 PyTorch embedding-space 방식으로 구현한다. 다만 CNN에서는 filter activation, max-pooling 위치, n-gram occlusion이 더 직관적이므로, IG는 Gradient x Input보다 안정적인 token attribution 보조 지표로 사용한다.

## 4. 노트북에 추가할 XAI 섹션 위치

추천 위치는 `best_cnn_model.pt`를 로드한 뒤이다.

현재 Cell 7은 다음 기능을 한 셀 안에 모두 포함한다.

```text
1. predict_sentiment 함수 정의
2. best_cnn_model.pt 로드
3. test set 평가
4. metrics.json에 test 결과 추가
5. ipywidgets UI 생성
```

XAI 분석 코드는 다음 둘 중 하나로 배치한다.

```text
권장안 A:
Cell 7을 실행한 뒤, 바로 아래에 "CNN XAI 분석" 섹션을 새로 추가한다.

권장안 B:
Cell 7을 나중에 정리할 때
"best model load + test eval" 셀과 "widget UI" 셀을 분리하고,
그 사이에 "CNN XAI 분석" 섹션을 넣는다.
```

반드시 `model.eval()` 상태에서 분석한다. Dropout이 켜져 있으면 같은 문장에 대해서도 설명 점수가 달라질 수 있다.

## 5. 공통 준비 함수 계획

모든 XAI 기법에서 같은 전처리와 예측 함수를 사용해야 결과가 일관된다.

### 5.1 분석용 디렉터리

분석 산출물은 다음 폴더에 저장한다.

```text
XAI/CNN/xai_outputs/
XAI/CNN/xai_outputs/figures/
```

노트북 실행 위치가 `XAI/CNN`일 수도 있고 repo root일 수도 있으므로, 처음에 경로를 확인하는 helper를 둔다.

```python
from pathlib import Path

if Path("nsmc_cnn.ipynb").exists() or Path("best_cnn_model.pt").exists():
    XAI_DIR = Path(".")
elif Path("XAI/CNN/best_cnn_model.pt").exists():
    XAI_DIR = Path("XAI/CNN")
else:
    XAI_DIR = Path(".")

OUTPUT_DIR = XAI_DIR / "xai_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
```

### 5.2 입력 인코딩 함수

현재 노트북의 `tokenize`, `word_to_index`, `max_len`을 그대로 사용한다.

```python
def encode_text_for_xai(text, word_to_index, max_len=30):
    tokens = tokenize(text)
    unk_idx = word_to_index.get("<unk>", 1)
    pad_idx = word_to_index.get("<pad>", 0)
    ids = [word_to_index.get(token, unk_idx) for token in tokens]
    original_len = min(len(ids), max_len)

    if len(ids) < max_len:
        ids = ids + [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        tokens = tokens[:max_len]

    return tokens, torch.LongTensor(ids), original_len
```

주의할 점은 다음과 같다.

```text
1. 분석 단위는 원문 단어가 아니라 Okt 형태소 token이다.
2. stopword로 제거된 단어는 XAI 결과에 나타나지 않는다.
3. max_len=30 뒤의 token은 모델 입력에 들어가지 않으므로 설명 대상이 아니다.
4. <unk>로 바뀐 token은 의미 해석이 약해질 수 있다.
```

### 5.3 예측 함수

XAI 점수 계산에서는 softmax 확률과 logit을 둘 다 저장한다.

```python
def predict_ids_for_xai(ids, model, device):
    model.eval()
    x = ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
    pred = int(probs.argmax().item())
    return {
        "logits": logits.squeeze(0).detach().cpu(),
        "probs": probs.detach().cpu(),
        "pred": pred,
        "neg_prob": float(probs[0].item()),
        "pos_prob": float(probs[1].item()),
    }
```

### 5.4 target class 선택 규칙

XAI 분석에서는 어떤 class를 설명할지 명확히 해야 한다.

```text
기본 분석:
target_class = model이 예측한 class

정답 근거 분석:
target_class = 실제 label

오분류 분석:
1. predicted class 기준으로 모델이 왜 틀렸는지 확인
2. true class 기준으로 어떤 근거가 부족했는지 확인
```

보고서에서는 predicted class 기준 결과를 기본으로 제시하고, FP/FN 사례에서는 predicted class와 true class 기준을 함께 비교한다.

## 6. 분석 샘플 선정 계획

전체 test set을 모두 자세히 설명하면 결과가 너무 커진다. 따라서 다음 방식으로 대표 샘플을 뽑는다.

### 6.1 정량 분석용 샘플

```text
test set에서 300개 샘플 무작위 추출
-> unigram occlusion, n-gram occlusion, max-pooling 통계에 사용
```

정량 분석에서는 다음을 집계한다.

```text
1. 긍정 예측에서 자주 중요한 token/n-gram
2. 부정 예측에서 자주 중요한 token/n-gram
3. filter size 3, 4, 5별 top activation n-gram
4. 오분류에서 자주 등장하는 혼동 n-gram
```

### 6.2 사례 분석용 샘플

보고서용으로 다음 네 종류를 각각 3개 이상 뽑는다.

```text
1. TP: 실제 긍정, 예측 긍정
2. TN: 실제 부정, 예측 부정
3. FP: 실제 부정, 예측 긍정
4. FN: 실제 긍정, 예측 부정
```

추가로 직접 만든 문장도 5개 정도 분석한다.

```text
1. 이 영화 진짜 시간 가는 줄 모르고 봤어요
2. 완전 최악이고 시간이 아까웠다
3. 배우는 좋았지만 스토리는 너무 지루했다
4. 생각보다 지루하지 않고 감동적이었다
5. 재미는 있는데 결말이 별로였다
```

이 문장들은 CNN이 단순 감정 단어뿐 아니라 연속 표현, 반전 표현, 부정어를 어떻게 처리하는지 확인하기 좋다.

## 7. Unigram Occlusion 계획

### 7.1 목적

Unigram occlusion은 token 하나를 가렸을 때 target class 확률이 얼마나 떨어지는지 측정한다.

CNN은 n-gram을 보는 모델이지만, unigram occlusion은 다음 기준선 역할을 한다.

```text
1. 개별 token 중 어떤 token이 예측에 가장 큰 영향을 주는가?
2. n-gram occlusion과 비교했을 때 CNN이 단어 단위보다 구 단위에 더 민감한가?
```

### 7.2 제거 방식

token을 실제로 삭제하면 뒤 token들의 위치가 당겨져 convolution window가 바뀐다. 그러면 "해당 token의 영향"과 "위치 이동의 영향"이 섞인다.

따라서 현재 CNN 코드에서는 삭제 대신 `<pad>` id로 치환하는 방식을 기본으로 한다.

```text
원본 ids:
[12, 57, 301, 9, 0, 0, ...]

2번 위치 occlusion:
[12, 57, 0, 9, 0, 0, ...]
```

### 7.3 점수 정의

기본 점수는 target class 확률 감소량이다.

```text
base_prob = P(target_class | original input)
masked_prob = P(target_class | token_i masked)
importance_i = base_prob - masked_prob
```

해석은 다음과 같다.

```text
importance_i > 0:
해당 token을 가리면 target class 확률이 떨어짐
-> target class 판단에 기여한 token

importance_i < 0:
해당 token을 가리면 target class 확률이 올라감
-> target class 판단을 방해했거나 반대 class 근거일 수 있음
```

확률은 softmax saturation 영향을 받을 수 있으므로 logit 감소량도 함께 저장한다.

```text
logit_importance_i = original_logit[target] - masked_logit[target]
```

### 7.4 산출물

```text
xai_outputs/cnn_unigram_occlusion.csv
xai_outputs/figures/unigram_occlusion_case_*.png
```

CSV 컬럼 예시는 다음과 같다.

| 컬럼 | 의미 |
|---|---|
| `sample_id` | test set index 또는 custom id |
| `text` | 원문 문장 |
| `tokens` | Okt token 목록 |
| `target_class` | 설명 대상 class |
| `pred_class` | 모델 예측 class |
| `base_prob` | 원본 target 확률 |
| `position` | token 위치 |
| `token` | 가린 token |
| `masked_prob` | 가린 뒤 target 확률 |
| `prob_drop` | target 확률 감소량 |
| `logit_drop` | target logit 감소량 |

### 7.5 보고서 해석 포인트

```text
Unigram occlusion 결과, 긍정 예측에서는 "재미있다", "감동", "최고"와 같은 token을 가렸을 때 긍정 확률이 크게 감소하였다.
부정 예측에서는 "지루하다", "최악", "아깝다"와 같은 token이 높은 중요도를 보였다.
다만 CNN은 convolution filter를 통해 연속 token을 함께 처리하므로, 단일 token occlusion만으로는 구 단위 감정 표현을 충분히 설명하기 어렵다.
```

## 8. n-gram Occlusion 계획

### 8.1 목적

n-gram occlusion은 CNN 분석에서 가장 중요한 입력 제거 기반 XAI이다. 현재 CNN은 `filter_sizes = [3, 4, 5]`를 사용하므로, 연속된 3, 4, 5개 token 구간이 모델 판단에 어떤 영향을 주는지 직접 확인할 수 있다.

비교를 위해 1-gram과 2-gram도 함께 계산한다.

```text
n = 1: unigram 기준선
n = 2: 짧은 감정 구
n = 3: filter size 3과 직접 대응
n = 4: filter size 4와 직접 대응
n = 5: filter size 5와 직접 대응
```

### 8.2 제거 방식

연속된 n개 token을 `<pad>`로 치환한다.

```text
tokens:
["배우", "좋다", "스토리", "너무", "지루하다"]

trigram occlusion at pos=2:
["배우", "좋다", "<pad>", "<pad>", "<pad>"]
```

### 8.3 점수 정의

Unigram occlusion과 동일하게 target class 확률 감소량을 사용한다.

```text
importance_window = P(target | original) - P(target | ngram masked)
```

CNN에서는 다음 해석이 가능하다.

```text
1. n-gram 점수가 unigram 점수보다 크면, 모델이 구 단위 패턴에 강하게 반응한 것이다.
2. filter size와 같은 길이의 n-gram에서 점수가 크게 나오면, 해당 kernel이 그 표현을 잡았을 가능성이 높다.
3. 긍정/부정 반전 문장에서는 어떤 구간이 최종 class를 뒤집었는지 확인할 수 있다.
```

### 8.4 산출물

```text
xai_outputs/cnn_ngram_occlusion.csv
xai_outputs/figures/ngram_occlusion_case_*.png
```

CSV 컬럼 예시는 다음과 같다.

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 샘플 id |
| `text` | 원문 |
| `target_class` | 설명 대상 class |
| `n` | occlusion window 크기 |
| `start_pos` | 시작 위치 |
| `end_pos` | 끝 위치 |
| `ngram` | 가린 token 구간 |
| `base_prob` | 원본 target 확률 |
| `masked_prob` | 가린 뒤 target 확률 |
| `prob_drop` | 중요도 |
| `logit_drop` | logit 기준 중요도 |

### 8.5 분석 표 예시

| rank | n | n-gram | target prob drop | 해석 |
|---:|---:|---|---:|---|
| 1 | 3 | `너무 지루하다` | 0.42 | 부정 판단 핵심 구간 |
| 2 | 2 | `시간 아깝다` | 0.35 | 부정 감정 구 |
| 3 | 1 | `지루하다` | 0.19 | 단일 부정 token |

### 8.6 보고서 해석 포인트

```text
n-gram occlusion 결과, CNN은 단일 token보다 "너무 지루하다", "시간 아깝다", "정말 최고다"처럼 연속된 감정 표현을 가렸을 때 예측 확률이 더 크게 변했다.
이는 현재 CNN이 filter size 3, 4, 5의 convolution을 통해 감정 n-gram 패턴을 학습했음을 보여준다.
```

## 9. Filter Activation Analysis 계획

### 9.1 목적

Filter activation analysis는 CNN 내부를 직접 보는 분석이다. 각 convolution filter가 어떤 n-gram에 강하게 반응했는지 확인한다.

현재 모델에는 총 300개 filter가 있다.

```text
filter size 3: 100개
filter size 4: 100개
filter size 5: 100개
```

각 filter는 입력 문장의 모든 가능한 n-gram 위치를 훑고, ReLU 이후 activation을 만든다. 이 activation이 클수록 해당 filter가 그 n-gram 패턴에 강하게 반응했다는 뜻이다.

### 9.2 현재 forward와 대응되는 내부 값

현재 코드의 forward는 다음 흐름이다.

```python
embedded = self.embedding(text)
embedded = embedded.unsqueeze(1)

for conv in self.convs:
    conved = F.relu(conv(embedded)).squeeze(3)
    pool = torch.max(conved, dim=2)[0]
```

여기서 분석 대상은 `conved`이다.

```text
conved shape:
[batch_size, n_filters, seq_len - filter_size + 1]

예:
filter size 3, max_len 30이면
[batch_size, 100, 28]
```

`conved[0, 12, 5]`는 다음 의미이다.

```text
0번 샘플에서
filter size 3 그룹의 12번 filter가
position 5부터 시작하는 trigram에 반응한 activation 값
```

### 9.3 filter별 top n-gram 추출

정량 분석용 300개 샘플에 대해 각 filter의 activation top-k n-gram을 저장한다.

```text
for each sample:
    for each conv block:
        conved = ReLU(conv(embedded))
        for each filter:
            top activation position 찾기
            position에 대응하는 token window 추출
            activation 값 저장
```

### 9.4 classifier weight까지 고려한 class 방향성

activation이 크다고 해서 무조건 긍정 근거는 아니다. 마지막 `fc` layer의 weight가 class 방향을 결정한다.

현재 feature concat 순서는 다음과 같다.

```text
filter size 3의 100개 pooled feature
filter size 4의 100개 pooled feature
filter size 5의 100개 pooled feature
```

따라서 feature index는 다음처럼 계산한다.

```text
conv_block_idx = 0 for filter size 3
conv_block_idx = 1 for filter size 4
conv_block_idx = 2 for filter size 5

feature_idx = conv_block_idx * n_filters + filter_idx
```

class 방향성은 다음처럼 본다.

```text
positive_direction = fc.weight[1, feature_idx] - fc.weight[0, feature_idx]

positive_direction > 0:
해당 filter activation은 긍정 class 쪽으로 기여

positive_direction < 0:
해당 filter activation은 부정 class 쪽으로 기여
```

샘플별 class 기여도는 다음처럼 계산한다.

```text
class_contribution = pooled_activation * fc.weight[target_class, feature_idx]
```

이 방식으로 "강하게 반응했지만 실제 class 결정에는 별로 기여하지 않는 filter"와 "activation도 크고 class weight도 커서 실제 판단에 기여한 filter"를 구분한다.

### 9.5 산출물

```text
xai_outputs/cnn_filter_top_ngrams.csv
xai_outputs/cnn_filter_class_direction.csv
xai_outputs/figures/filter_activation_top_patterns.png
```

`cnn_filter_top_ngrams.csv` 컬럼 예시는 다음과 같다.

| 컬럼 | 의미 |
|---|---|
| `filter_size` | 3, 4, 5 |
| `filter_idx` | 해당 conv block 안의 filter 번호 |
| `feature_idx` | fc 입력 feature 번호 |
| `sample_id` | 샘플 id |
| `text` | 원문 |
| `true_label` | 실제 label |
| `pred_label` | 예측 label |
| `activation` | ReLU activation |
| `position` | activation 위치 |
| `ngram` | 해당 위치 token window |
| `positive_direction` | 긍정 class 방향성 |
| `target_contribution` | target class 기여도 |

### 9.6 보고서 해석 포인트

```text
Filter activation 분석 결과, 일부 filter는 "정말 재밌다", "완전 감동"과 같은 긍정 n-gram에서 반복적으로 높은 activation을 보였고,
다른 filter는 "시간 아깝다", "너무 지루하다"와 같은 부정 n-gram에서 높은 activation을 보였다.
이는 CNN이 개별 단어가 아니라 연속된 형태소 패턴을 감정 판단 단위로 사용한다는 것을 보여준다.
```

## 10. Max-pooling 위치 분석 계획

### 10.1 목적

현재 CNN은 각 filter마다 sequence dimension에서 max-pooling을 수행한다.

```text
pool = torch.max(conved, dim=2)[0]
```

이때 실제로는 max value뿐 아니라 max index도 얻을 수 있다.

```python
pool_values, pool_indices = torch.max(conved, dim=2)
```

`pool_indices`는 각 filter가 문장에서 가장 강하게 반응한 위치이다. 이 위치를 token window로 바꾸면 "각 filter가 최종적으로 선택한 n-gram"을 확인할 수 있다.

### 10.2 분석 방식

개별 문장에 대해 다음을 추출한다.

```text
1. 각 filter size별 pooled activation
2. 각 filter별 max-pooling position
3. position에 해당하는 n-gram
4. fc weight를 고려한 target class contribution
5. contribution 기준 top-k filter와 n-gram
```

최종적으로 문장 하나당 다음 표를 만든다.

| rank | filter size | filter idx | selected n-gram | activation | class contribution | 방향 |
|---:|---:|---:|---|---:|---:|---|
| 1 | 3 | 27 | `너무 지루하다` | 4.12 | 1.83 | 부정 |
| 2 | 4 | 8 | `스토리 너무 지루하다` | 3.77 | 1.45 | 부정 |
| 3 | 5 | 91 | `배우 좋다 스토리 너무 지루하다` | 2.91 | 0.88 | 부정 |

### 10.3 산출물

```text
xai_outputs/cnn_maxpool_positions.csv
xai_outputs/figures/maxpool_case_*.png
```

CSV 컬럼 예시는 다음과 같다.

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 샘플 id |
| `text` | 원문 |
| `target_class` | 설명 class |
| `filter_size` | 3, 4, 5 |
| `filter_idx` | filter 번호 |
| `max_position` | max-pooling 위치 |
| `selected_ngram` | 선택된 n-gram |
| `activation` | max activation |
| `target_fc_weight` | target class fc weight |
| `target_contribution` | activation x fc weight |

### 10.4 보고서 해석 포인트

```text
Max-pooling 위치 분석에서는 각 filter가 문장 전체에서 하나의 핵심 위치를 선택하는 과정을 확인하였다.
예를 들어 부정 예측 문장에서 target contribution이 큰 filter들은 "너무 지루하다", "시간 아깝다"와 같은 구간을 max-pooling 위치로 선택하였다.
이는 CNN의 최종 판단이 여러 filter가 선택한 감정 n-gram feature의 조합으로 이루어짐을 보여준다.
```

## 11. Gradient x Input 계획

### 11.1 목적

Gradient x Input은 target logit이 embedding input에 얼마나 민감한지 보는 attribution 방법이다. CNN에서는 convolution 때문에 중요도가 주변 token에 퍼질 수 있으므로, occlusion과 함께 보조적으로 사용한다.

분석 목적은 다음과 같다.

```text
1. token-level attribution heatmap 생성
2. n-gram occlusion 결과와 gradient 결과가 비슷한지 확인
3. 오분류 사례에서 모델이 민감하게 본 token 확인
```

### 11.2 구현 방식

현재 `CNN_Sentiment.forward()`는 token id를 입력받아 내부에서 embedding을 계산한다. Gradient x Input을 계산하려면 embedding tensor의 gradient를 잡아야 한다.

가장 단순한 방법은 XAI용 forward 함수를 따로 작성하는 것이다.

```python
def forward_from_embedded_for_xai(model, embedded):
    embedded = embedded.unsqueeze(1)
    pooled = []
    for conv in model.convs:
        conved = F.relu(conv(embedded)).squeeze(3)
        pool = torch.max(conved, dim=2)[0]
        pooled.append(pool)
    cat = torch.cat(pooled, dim=1)
    return model.fc(cat)
```

dropout은 XAI에서 끄는 것이 좋으므로 `model.eval()` 상태에서 dropout을 적용하지 않는다. 위 함수는 dropout을 생략한다. 현재 dropout은 학습 정규화 목적이므로, 설명에서는 deterministic한 결과를 우선한다.

계산 흐름은 다음과 같다.

```python
ids = ids.unsqueeze(0).to(device)
embedded = model.embedding(ids)
embedded.retain_grad()

logits = forward_from_embedded_for_xai(model, embedded)
target_logit = logits[0, target_class]

model.zero_grad()
target_logit.backward()

grad = embedded.grad[0]
emb = embedded.detach()[0]
scores = (grad * emb).sum(dim=1)
```

점수는 두 가지로 저장한다.

```text
signed_score:
target class 방향으로 미는지, 반대로 미는지 확인

abs_score:
중요도 크기만 확인
```

보고서 그림에는 `abs_score`를 기본으로 쓰고, 해석에는 signed score를 참고한다.

### 11.3 산출물

```text
xai_outputs/cnn_gradient_x_input.csv
xai_outputs/figures/gradient_x_input_case_*.png
```

CSV 컬럼 예시는 다음과 같다.

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 샘플 id |
| `text` | 원문 |
| `target_class` | 설명 대상 class |
| `position` | token 위치 |
| `token` | 형태소 token |
| `signed_score` | Gradient x Input signed attribution |
| `abs_score` | 절댓값 중요도 |
| `normalized_abs_score` | 0-1 정규화 점수 |

### 11.4 해석 주의점

```text
1. Gradient x Input은 local sensitivity이므로 실제 token 제거 효과와 다를 수 있다.
2. CNN에서는 filter window 때문에 인접 token들이 함께 높은 점수를 받을 수 있다.
3. 확률 변화 기반인 occlusion과 함께 비교해야 설득력이 높다.
4. pad token은 결과에서 제외한다.
```

## 12. Integrated Gradients 구현 계획

Integrated Gradients는 Gradient x Input과 같은 gradient 기반 attribution이지만, 현재 입력 한 점의 gradient만 보는 대신 baseline부터 실제 입력까지의 경로를 따라 gradient를 누적한다.

현재 CNN에서는 baseline을 `<pad>` sequence로 둘 수 있다.

```text
baseline ids:
[pad, pad, pad, ..., pad]

input ids:
[token1, token2, ..., pad, pad]
```

다만 token id는 discrete 값이므로, 실제 IG는 embedding space에서 계산한다.

```text
baseline embedding = embedding(pad ids)
input embedding = embedding(input ids)
두 embedding 사이를 m step으로 선형 보간
각 step에서 target logit gradient 계산
평균 gradient x embedding difference
```

현재 프로젝트 구현은 Captum 없이 PyTorch만 사용한다.

```text
구현 파일:
XAI/CNN/xai_methods/integrated_gradients.py

출력 파일:
XAI/CNN/xai_outputs/cnn_integrated_gradients.csv

확인 notebook:
XAI/CNN/notebooks/06_cnn_integrated_gradients.ipynb
```

보고서에서는 다음처럼 설명한다.

```text
Integrated Gradients는 `<pad>` baseline에서 실제 입력 embedding까지의 경로를 따라 target class logit의 gradient를 누적하여 token별 attribution을 계산한다. 본 프로젝트에서는 Gradient x Input과 함께 CNN의 token-level 보조 설명 지표로 사용하였다.
```

## 13. Grad-CAM for Text CNN 선택 계획

Grad-CAM은 이미지 CNN에서 직관적이지만, 현재 텍스트 CNN에서는 필수로 구현하지 않는다.

제외 이유는 다음과 같다.

```text
1. 현재 모델은 Conv2d + global max-pooling 구조라 filter activation과 max-pooling 위치가 더 직접적이다.
2. 텍스트 입력은 이미지처럼 2D 영역이 아니라 형태소 sequence이다.
3. 발표/보고서에서는 n-gram occlusion과 filter activation이 더 설명하기 쉽다.
```

따라서 Grad-CAM은 확장 가능 기법으로만 언급한다.

## 14. 시각화 계획

### 14.1 token heatmap

Unigram occlusion과 Gradient x Input은 token별 heatmap으로 보여준다.

표시 방식:

```text
token 아래에 score 표시
중요도가 높은 token은 진한 색
부정 방향/긍정 방향을 색상으로 구분
```

예시:

| token | 배우 | 좋다 | 스토리 | 너무 | 지루하다 |
|---|---:|---:|---:|---:|---:|
| occlusion drop | 0.02 | -0.05 | 0.01 | 0.18 | 0.39 |

### 14.2 n-gram bar chart

n-gram occlusion top-k는 horizontal bar chart로 표시한다.

```text
y축: n-gram
x축: target probability drop
색상: n 값 또는 class
```

### 14.3 filter activation table

Filter activation은 그림보다 표가 더 명확하다.

```text
filter size / filter idx / top n-gram / activation / class direction
```

같은 filter가 여러 샘플에서 비슷한 표현에 반응하면, 해당 filter를 감정 phrase detector처럼 해석할 수 있다.

### 14.4 max-pooling 위치 강조

개별 사례에서는 target contribution이 높은 filter top 5만 보여준다.

```text
원문:
배우는 좋았지만 스토리는 너무 지루했다.

CNN이 선택한 주요 n-gram:
1. "너무 지루하다" - filter size 3, 부정 기여
2. "스토리 너무 지루하다" - filter size 4, 부정 기여
3. "배우 좋다" - filter size 3, 긍정 기여지만 최종 부정 기여보다 작음
```

## 15. 최종 산출물 목록

최종적으로 생성할 파일은 다음과 같다.

```text
XAI/CNN/xai_outputs/cnn_xai_selected_samples.csv
XAI/CNN/xai_outputs/cnn_unigram_occlusion.csv
XAI/CNN/xai_outputs/cnn_ngram_occlusion.csv
XAI/CNN/xai_outputs/cnn_filter_top_ngrams.csv
XAI/CNN/xai_outputs/cnn_filter_class_direction.csv
XAI/CNN/xai_outputs/cnn_maxpool_positions.csv
XAI/CNN/xai_outputs/cnn_gradient_x_input.csv
XAI/CNN/xai_outputs/cnn_integrated_gradients.csv
XAI/CNN/xai_outputs/cnn_xai_case_summary.md

XAI/CNN/xai_outputs/figures/unigram_occlusion_case_*.png
XAI/CNN/xai_outputs/figures/ngram_occlusion_case_*.png
XAI/CNN/xai_outputs/figures/gradient_x_input_case_*.png
XAI/CNN/xai_outputs/figures/integrated_gradients_case_*.png
XAI/CNN/xai_outputs/figures/filter_activation_top_patterns.png
XAI/CNN/xai_outputs/figures/maxpool_case_*.png
```

보고서에는 모든 CSV를 다 넣지 않고, 핵심 표와 그림만 선별해서 사용한다.

## 16. 단계별 실행 계획

### Step 1. XAI 준비 셀 추가

목표:

```text
1. model.eval()
2. best_cnn_model.pt 로드 확인
3. OUTPUT_DIR 생성
4. label 이름 정의
5. encode/predict helper 정의
```

완료 기준:

```text
직접 입력 문장 1개에 대해 tokens, ids, pred, pos_prob, neg_prob가 출력된다.
```

### Step 2. test set 예측 결과 수집

목표:

```text
test_loader 또는 test_data 기준으로 전체 test prediction을 만든다.
TP/TN/FP/FN을 구분한다.
confidence가 높은 샘플과 낮은 샘플을 분리한다.
```

완료 기준:

```text
cnn_xai_selected_samples.csv 생성
TP/TN/FP/FN 사례가 각각 최소 3개 이상 확보된다.
```

### Step 3. Unigram Occlusion 구현

목표:

```text
각 샘플에서 token 하나씩 <pad>로 바꾼다.
target class 확률과 logit 감소량을 계산한다.
top-k token을 저장한다.
```

완료 기준:

```text
cnn_unigram_occlusion.csv 생성
case별 token importance 표와 bar chart 생성
```

### Step 4. n-gram Occlusion 구현

목표:

```text
n = 1, 2, 3, 4, 5에 대해 연속 token window를 <pad>로 바꾼다.
filter size 3, 4, 5와 직접 비교한다.
```

완료 기준:

```text
cnn_ngram_occlusion.csv 생성
각 case에서 top n-gram이 감정 표현과 연결되는지 확인
```

### Step 5. Filter Activation 구현

목표:

```text
model.convs를 직접 통과시켜 conved activation을 얻는다.
filter별 top activation n-gram을 저장한다.
fc weight로 긍정/부정 방향성을 계산한다.
```

완료 기준:

```text
cnn_filter_top_ngrams.csv 생성
cnn_filter_class_direction.csv 생성
긍정 방향 filter와 부정 방향 filter 예시를 각각 5개 이상 확보
```

### Step 6. Max-pooling 위치 분석 구현

목표:

```text
torch.max(conved, dim=2)의 indices를 저장한다.
각 filter가 선택한 token window를 복원한다.
class contribution 기준 top-k filter를 뽑는다.
```

완료 기준:

```text
cnn_maxpool_positions.csv 생성
개별 문장 5개 이상에 대해 "CNN이 선택한 핵심 n-gram" 표 작성
```

### Step 7. Gradient x Input 구현

목표:

```text
embedding gradient를 계산한다.
token별 signed score와 abs score를 저장한다.
occlusion 결과와 비교한다.
```

완료 기준:

```text
cnn_gradient_x_input.csv 생성
case별 token attribution heatmap 생성
```

### Step 8. 사례 분석 정리

목표:

```text
TP/TN/FP/FN 각 사례에서
1. 예측 확률
2. top unigram occlusion
3. top n-gram occlusion
4. max-pooling selected n-gram
5. gradient attribution
을 한 페이지 단위로 정리한다.
```

완료 기준:

```text
cnn_xai_case_summary.md 생성
보고서에 넣을 대표 사례 4개 이상 선정
```

## 17. CNN 전용 보고서 구조

최종 보고서는 CNN만 다루므로 다음 구조가 적절하다.

```text
1. CNN 모델 개요
   - Okt 형태소 분석
   - vocabulary와 max_len=30
   - embedding + Conv2d filter size 3/4/5 + max-pooling 구조
   - test accuracy 약 84.95%

2. XAI 분석 목표
   - CNN이 어떤 형태소 n-gram을 보는지 확인
   - filter activation과 max-pooling 위치를 통해 내부 판단 근거 분석

3. Occlusion 분석
   - unigram occlusion 결과
   - n-gram occlusion 결과
   - n-gram 단위 제거가 CNN 설명에 더 적합한 이유

4. Filter Activation 분석
   - 긍정 방향 filter 예시
   - 부정 방향 filter 예시
   - filter size별 top n-gram 패턴

5. Max-pooling 위치 분석
   - 각 filter가 선택한 핵심 n-gram
   - class contribution 기준 top-k evidence

6. Gradient x Input 분석
   - token attribution heatmap
   - occlusion 결과와 비교

7. 성공/실패 사례 분석
   - TP, TN 예시
   - FP, FN 예시
   - 오분류 원인 해석

8. 한계
   - 형태소 token 기준 설명의 한계
   - stopword 제거로 사라진 표현
   - max_len=30 truncation
   - gradient 기반 설명의 불안정성
   - XAI는 모델 판단의 근사 설명이라는 점

9. 결론
   - CNN은 단일 단어보다 연속된 감정 n-gram에 강하게 반응한다.
   - filter activation과 max-pooling 위치 분석은 TextCNN의 판단 근거를 직관적으로 보여준다.
   - n-gram occlusion은 현재 CNN 구조에 가장 잘 맞는 입력 변화 기반 설명 방법이다.
```

## 18. 결과 해석 문장 예시

보고서에 바로 사용할 수 있는 문장 예시는 다음과 같다.

```text
본 CNN 모델은 Okt 형태소 단위로 토큰화된 NSMC 리뷰를 입력받고, filter size 3, 4, 5의 convolution filter를 통해 연속된 형태소 패턴을 추출한다.
따라서 CNN의 설명에서는 단일 token보다 n-gram 단위의 감정 표현을 확인하는 것이 중요하다.
```

```text
n-gram occlusion 결과, "너무 지루하다", "시간 아깝다"와 같은 부정 표현 구간을 가렸을 때 부정 class 확률이 크게 감소하였다.
이는 CNN이 개별 단어가 아니라 연속된 감정 표현을 하나의 판단 근거로 사용했음을 보여준다.
```

```text
Filter activation 분석에서는 특정 filter가 여러 부정 리뷰에서 반복적으로 "최악", "시간 아깝다", "너무 지루하다"와 유사한 n-gram에 강하게 반응하였다.
마지막 classifier의 class weight까지 함께 확인한 결과, 해당 filter activation은 부정 class logit 증가에 기여하는 것으로 해석된다.
```

```text
Max-pooling 위치 분석은 각 filter가 문장 전체 중 어느 위치를 최종 feature로 선택했는지 보여준다.
오분류 사례에서는 긍정 표현 일부가 높은 activation을 얻어 최종 긍정 예측에 기여했지만, 문장 후반의 부정 반전 표현을 충분히 반영하지 못한 경우가 확인될 수 있다.
```

```text
Gradient x Input 결과는 token별 민감도를 제공하지만, CNN에서는 convolution window 때문에 중요도가 인접 token으로 퍼지는 경향이 있다.
따라서 본 분석에서는 gradient attribution을 보조 지표로 사용하고, 최종 해석은 n-gram occlusion 및 filter activation 결과와 함께 비교하였다.
```

## 19. 체크리스트

구현 전 확인:

```text
[ ] best_cnn_model.pt가 현재 노트북의 model 구조와 일치한다.
[ ] word_to_index, index_to_word, tokenize가 현재 세션에 존재한다.
[ ] max_len이 학습 때와 동일하게 30이다.
[ ] model.eval() 상태이다.
[ ] device가 cuda/cpu 중 하나로 정상 설정되어 있다.
```

분석 실행 후 확인:

```text
[ ] unigram occlusion에서 pad token은 제외했다.
[ ] n-gram occlusion에서 실제 token 길이를 넘는 window는 제외했다.
[ ] filter activation에서 filter size별 window 길이가 정확히 맞는다.
[ ] max-pooling index를 token window로 복원할 때 position offset이 맞는다.
[ ] fc weight 방향성을 함께 고려했다.
[ ] target class가 predicted class인지 true class인지 결과 파일에 기록했다.
[ ] 오분류 사례는 predicted 기준과 true 기준 설명을 구분했다.
```

보고서 작성 전 확인:

```text
[ ] CNN 외 모델 분석 내용이 섞이지 않았다.
[ ] Attention, CLS 등 Transformer 용어를 사용하지 않았다.
[ ] 형태소 token 기준 분석임을 명시했다.
[ ] XAI 결과를 모델 판단의 완전한 증명이 아니라 근사 설명으로 표현했다.
[ ] 대표 사례마다 원문, token, 예측 확률, top evidence를 함께 제시했다.
```

## 20. 최종 결론 방향

이 CNN XAI 분석의 결론은 다음 방향으로 정리한다.

```text
현재 CNN 모델은 NSMC 리뷰를 형태소 token sequence로 입력받아,
filter size 3, 4, 5의 convolution filter로 연속된 감정 표현을 탐지한다.

Unigram occlusion은 개별 token의 영향을 보여주는 기준선 역할을 하지만,
CNN 구조를 가장 잘 설명하는 것은 n-gram occlusion, filter activation, max-pooling 위치 분석이다.

특히 n-gram occlusion과 filter activation 결과가 같은 표현 구간을 가리킬 경우,
해당 n-gram은 CNN 예측의 강한 근거로 해석할 수 있다.

오분류 사례에서는 CNN이 특정 감정 n-gram에는 강하게 반응했지만,
반전 표현이나 문장 전체 문맥을 충분히 반영하지 못한 한계를 확인할 수 있다.
```

## 21. 구현 파일 및 실행 명령

CNN XAI 분석 코드는 노트북과 분리하여 다음 파일에 구현한다.

```text
XAI/CNN/nsmc_cnn_xai.py
```

전체 분석 실행 명령은 repo root 기준 다음과 같다.

```powershell
python XAI\CNN\nsmc_cnn_xai.py --sample-size 300 --pool-size 1000 --output-dir XAI\CNN\xai_outputs
```

처음 실행하면 노트북과 동일한 방식으로 train 데이터를 Okt 토큰화하여 vocabulary cache를 만든다. 이후 실행부터는 다음 캐시를 재사용한다.

```text
XAI/CNN/cnn_preprocess_cache.pkl
```

전처리 캐시를 다시 만들고 싶으면 다음처럼 실행한다.

```powershell
python XAI\CNN\nsmc_cnn_xai.py --refresh-cache --sample-size 300 --pool-size 1000 --output-dir XAI\CNN\xai_outputs
```

빠른 동작 확인만 하고 싶으면 직접 입력 문장만 분석한다.

```powershell
python XAI\CNN\nsmc_cnn_xai.py --only-custom --sample-size 0 --pool-size 0 --output-dir XAI\CNN\xai_outputs_smoke
```

현재 구현 스크립트가 생성하는 산출물은 다음과 같다.

```text
XAI/CNN/xai_outputs/cnn_xai_selected_samples.csv
XAI/CNN/xai_outputs/cnn_unigram_occlusion.csv
XAI/CNN/xai_outputs/cnn_ngram_occlusion.csv
XAI/CNN/xai_outputs/cnn_filter_top_ngrams.csv
XAI/CNN/xai_outputs/cnn_filter_class_direction.csv
XAI/CNN/xai_outputs/cnn_maxpool_positions.csv
XAI/CNN/xai_outputs/cnn_gradient_x_input.csv
XAI/CNN/xai_outputs/cnn_integrated_gradients.csv
XAI/CNN/xai_outputs/cnn_xai_case_summary.md
XAI/CNN/xai_outputs/cnn_xai_run_metadata.json
XAI/CNN/xai_outputs/figures/*.png
```

코드 분리, 방법별 notebook 구성, FNN/CNN/Transformer/LLM 비교까지 확장하는 개선 계획은 다음 문서에 정리한다.

```text
XAI/XAI_CODE_SPLIT_AND_LLM_COMPARISON_PLAN.md
```
