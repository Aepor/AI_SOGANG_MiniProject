# Sentiment Direction XAI Experiment Plan

## 1. 실험 목표

이 실험의 목표는 Transformer 감정분석 모델 내부에 긍정/부정을 나타내는 선형 감정 방향이 존재하는지 확인하고, XAI가 중요하다고 판단한 단어를 제거했을 때 문장 hidden state가 특히 그 감정 방향으로 움직이는지 검증하는 것이다.

핵심 질문은 두 개로 나눈다.

1. Transformer의 `[CLS]` hidden state 공간에 감정 방향 `v_sentiment`가 존재하는가?
2. XAI 중요 단어를 제거하면 hidden state 변화가 단순히 아무 방향으로 흔들리는 것이 아니라, 원래 감정을 약화시키는 `v_sentiment` 방향으로 정렬되는가?

따라서 이 실험은 단순히 "중요 단어 제거 후 예측 확률이 변했다"를 보는 것이 아니라, 모델 내부 표현 변화의 방향까지 분석한다.

## 2. 핵심 아이디어

Transformer 분류 모델에서 각 encoder layer의 `[CLS]` hidden state를 문장 표현 후보로 둔다. 마지막 layer 하나만 보지 않고, embedding output부터 마지막 encoder layer까지 모두 비교한다.

```text
h_l(x) = layer l의 CLS hidden state of sentence x
```

아래 수식은 특정 layer `l` 하나를 기준으로 쓴다. 훈련 데이터의 긍정 문장 hidden state 평균과 부정 문장 hidden state 평균을 계산한다.

```text
mu_pos_l = mean(h_l(x) for positive train samples)
mu_neg_l = mean(h_l(x) for negative train samples)
v_sentiment_l = mu_pos_l - mu_neg_l
v_l = v_sentiment_l / ||v_sentiment_l||
center_l = (mu_pos_l + mu_neg_l) / 2
```

`v_l`은 해당 layer에서 부정 중심에서 긍정 중심으로 향하는 단위 방향 벡터다. 문장 `x`의 projection score는 다음처럼 정의한다.

```text
projection_l(x) = (h_l(x) - center_l) dot v_l
```

해석:

- `projection(x) > 0`: hidden state가 긍정 중심 쪽에 더 가까움
- `projection(x) < 0`: hidden state가 부정 중심 쪽에 더 가까움
- `abs(projection(x))`가 클수록 감정 방향성이 강함

### 2.1 Hidden layer 선택 원칙

마지막 layer의 CLS hidden state는 classifier head 바로 앞에 있다. 따라서 마지막 layer에서 sentiment direction이 잘 나오는 것은 어느 정도 예상 가능한 결과다. 이 결과만으로는 "모델 내부 깊은 representation에 감정 개념이 있다"고 강하게 주장하기 어렵다.

이 비판을 피하기 위해 다음 원칙을 둔다.

1. 모든 layer에서 `v_sentiment_l`을 따로 계산한다.
2. 마지막 layer는 classifier-proximal reference로 보고, 단독 근거로 쓰지 않는다.
3. primary analysis는 중간-후반 encoder layer에서 수행한다.
4. final layer 결과는 성능 상한 또는 classifier 직전 표현의 참고 결과로 해석한다.

KcELECTRA/BERT 계열에서 hidden state index는 보통 다음처럼 해석한다.

```text
hidden_states[0]  = embedding output
hidden_states[1]  = encoder layer 1 output
...
hidden_states[12] = encoder layer 12 output
```

권장 primary layer:

```text
primary: layer 8 또는 layer 9
secondary: all layers 0~12
reference: final layer 12
```

다만 모델별 layer 수가 다를 수 있으므로 구현에서는 `model.config.num_hidden_layers`로 마지막 layer index를 확인한다. 최종 보고서에서는 layer 8/9를 고정으로만 주장하지 않고, 전체 layer curve를 함께 제시한다.

## 3. 가설

### H1. Sentiment Direction 존재 가설

`v_sentiment`가 실제 감정 정보를 담고 있다면, validation/test 문장에서 projection score만으로도 긍정/부정을 어느 정도 구분할 수 있어야 한다.

검증:

- positive sample의 projection 평균이 negative sample보다 큰가?
- `projection > 0` 기준 label accuracy가 높은가?
- projection score의 ROC-AUC가 높은가?
- projection score와 model logit margin이 양의 상관을 갖는가?

```text
logit_margin = logit_positive - logit_negative
correlation(projection, logit_margin)
```

### H2. Linear Probe 검증 가설

CLS hidden state에 감정 정보가 선형적으로 들어 있다면, hidden state 위에 간단한 logistic regression probe를 학습했을 때 좋은 성능이 나와야 한다.

검증:

- `LogisticRegression(h -> label)` validation/test accuracy, F1, ROC-AUC
- probe weight와 `v_sentiment`의 cosine similarity

```text
cosine(w_probe, v_sentiment)
```

해석:

- Projection과 Linear Probe가 모두 좋음: 감정 정보가 거의 하나의 선형 방향으로 잘 정리되어 있음
- Probe는 좋은데 Projection은 약함: 감정 정보는 있지만 단순 평균 차이 방향 하나로는 부족함
- 둘 다 약함: 해당 layer의 CLS가 감정 정보를 잘 담지 못했거나 checkpoint/데이터 문제가 있음

### H3. XAI 중요 단어 제거 후 감정 약화 가설

XAI가 중요하다고 고른 단어가 실제로 모델 내부 감정 표현을 지탱한다면, 그 단어를 제거하거나 `[MASK]`로 바꿨을 때 projection이 원래 예측 감정을 약화시키는 방향으로 움직여야 한다.

모델 예측 기준 sign을 둔다.

```text
y_pred = +1 if model predicts positive else -1
strength(x) = y_pred * projection(x)
```

원본 문장과 제거 문장을 비교한다.

```text
h0 = h(original)
h1 = h(removed_or_masked)
delta_h = h1 - h0
delta_projection = delta_h dot v
sentiment_weakening = -y_pred * delta_projection
```

해석:

- `sentiment_weakening > 0`: 제거 후 원래 예측 감정 방향성이 약해짐
- `sentiment_weakening = 0`: 감정축 위 변화가 거의 없음
- `sentiment_weakening < 0`: 오히려 원래 감정 방향성이 강해짐

### H4. 방향 특이성 가설

XAI 단어 제거 후 hidden state가 많이 변했다는 것만으로는 부족하다. 변화 벡터 `delta_h`가 특히 sentiment direction으로 정렬되는지 확인해야 한다.

`v`가 단위 벡터일 때 sentiment 방향 성분은 다음과 같다.

```text
delta_parallel = (delta_h dot v) * v
delta_orthogonal = delta_h - delta_parallel
```

방향 특이성 지표:

```text
delta_norm = ||delta_h||
parallel_norm = abs(delta_h dot v)
orthogonal_norm = ||delta_orthogonal||

sentiment_ratio = parallel_norm / (delta_norm + eps)
alignment = cosine(delta_h, -y_pred * v)
```

해석:

- `sentiment_ratio`가 큼: 전체 hidden state 변화 중 sentiment 축 성분 비율이 큼
- `alignment`가 양수로 큼: 변화 방향이 원래 예측 감정을 약화시키는 방향과 일치함
- `alignment`가 0 근처: sentiment 방향과 무관한 변화
- `alignment`가 음수: 원래 예측 감정을 더 강화하는 방향

### H5. Classifier-proximity 통제 가설

마지막 layer에서만 좋은 결과가 나오면, 그 결과는 classifier head 바로 앞의 task-specific feature를 본 것일 수 있다. 따라서 이 실험은 마지막 layer 결과를 성능 상한으로 보되, 중간-후반 layer에서도 비슷한 현상이 유지되는지 확인한다.

검증:

- layer별 projection ROC-AUC curve
- layer별 linear probe 성능 curve
- layer별 XAI ablation `delta_strength`, `alignment`, `sentiment_ratio`

해석:

- 중간-후반 layer에서도 projection/probe/ablation alignment가 좋음: 감정 정보가 classifier 직전뿐 아니라 encoder 내부 representation에도 형성되어 있다는 강한 근거
- 마지막 layer에서만 좋음: 감정 정보가 주로 classifier 직전 task-specific 표현에 집중되어 있다는 제한적 결론
- 초반 layer부터 좋음: lexical/subword 수준의 감정 단서가 매우 강하거나 데이터셋 shortcut이 있을 수 있으므로 추가 확인 필요

## 4. 데이터와 분할

### 4.1 Direction 생성 데이터

`v_sentiment`는 반드시 train split에서만 만든다.

권장:

- `Data/NSMC/ratings_train.txt`를 train/validation으로 분할
- 기존 Transformer 학습과 동일하게 train 내부에서 validation ratio 0.1 사용
- `v_sentiment` 계산에는 train 부분만 사용
- layer 선택, threshold 선택, probe 튜닝은 validation에서 수행
- 최종 수치는 test 또는 selected review set에서 보고

주의:

- `ratings_test.txt` 또는 발표용 selected samples로 `v_sentiment`를 만들면 leakage가 생긴다.
- positive/negative 수가 다르면 class별 평균은 각각 독립적으로 계산하되, 필요하면 class-balanced sampling도 함께 비교한다.

### 4.2 분석 대상 샘플

최종 qualitative 분석은 다음 sample set에 적용한다.

- Opus/XAI 비교용 selected reviews
- Transformer가 맞춘 샘플과 틀린 샘플
- 명확한 긍정, 명확한 부정, 긍정+부정 혼합, 반전 표현 샘플

샘플별로 true label, model prediction, confidence를 함께 저장한다.

## 5. 모델과 hidden state 추출

기본 모델은 현재 Transformer 스크립트의 저장 모델 경로를 따른다.

```text
XAI/Transformer/kcelectra_nsmc_model
```

단, checkpoint는 Git에 포함되지 않을 수 있으므로 실행 전 실제 model directory 존재 여부를 확인한다. 없으면 기존 학습 스크립트로 재학습하거나 외부 checkpoint 경로를 `--model-dir`로 지정한다.

hidden state 추출 시 Hugging Face 모델 호출에 다음 옵션을 사용한다.

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
)
hidden_states = outputs.hidden_states
```

layer별 `[CLS]` 추출:

```text
h_layer_l = hidden_states[l][:, 0, :]
```

layer index는 코드에서 자동으로 확인한다.

```python
num_encoder_layers = model.config.num_hidden_layers
layer_indices = range(num_encoder_layers + 1)
final_layer = num_encoder_layers
primary_candidate_layers = [final_layer - 4, final_layer - 3]
```

기본 분석은 모든 layer를 측정한다. 마지막 layer는 classifier head 바로 앞이므로 reference로 보고, primary analysis는 중간-후반 layer를 중심으로 한다. KcELECTRA-base처럼 encoder layer가 12개인 경우에는 layer 8 또는 layer 9를 primary 후보로 둔다.

결과 해석 원칙:

- final layer만 좋음: classifier-proximal feature를 본 제한적 결과
- middle/late layer도 좋음: encoder 내부 representation에 감정 방향이 형성된 강한 결과
- early layer도 좋음: lexical shortcut 또는 감정 단어 표면 단서 가능성 추가 점검

출력에는 `layer_role`을 함께 저장한다.

```text
embedding: hidden_states[0]
early: encoder 앞쪽 layer
middle: encoder 중간 layer
primary_candidate: 중간-후반 primary 분석 후보
final_reference: classifier head 바로 앞 마지막 layer
```

## 6. Projection Score 검증 절차

각 layer `l`에 대해 다음을 계산한다.

```text
mu_pos_l
mu_neg_l
v_l
center_l
projection_l(x)
```

출력 파일:

```text
XAI/outputs_sentiment_direction/
  direction_layer_00.npz
  direction_layer_01.npz
  ...
  projection_scores.csv
  layer_projection_summary.csv
```

`projection_scores.csv` 컬럼:

```text
sample_id
split
text
true_label
prediction
probability
layer
layer_role
projection
strength_by_true_label
strength_by_prediction
logit_negative
logit_positive
logit_margin
```

`layer_projection_summary.csv` 컬럼:

```text
layer
layer_role
n_samples
mean_projection_positive
mean_projection_negative
projection_gap
threshold_zero_accuracy
roc_auc
spearman_with_logit_margin
pearson_with_logit_margin
```

## 7. Linear Probe 검증 절차

각 layer의 CLS hidden state를 입력으로 logistic regression을 학습한다.

```text
train: probe 학습
validation: C 등 hyperparameter 선택
test: 최종 성능 보고
```

모델 본체는 절대 업데이트하지 않는다. Probe는 representation에 감정 정보가 선형적으로 들어 있는지 확인하는 도구다.

출력 파일:

```text
linear_probe_metrics.csv
linear_probe_weights_layer_XX.npz
```

`linear_probe_metrics.csv` 컬럼:

```text
layer
layer_role
accuracy
precision
recall
f1
roc_auc
cosine_probe_weight_with_v_sentiment
```

## 8. XAI 중요 단어 선정

Transformer XAI 결과는 단어 또는 eojeol 단위로 정렬한다. Transformer token은 subword이므로 human-facing 분석은 eojeol 단위가 좋고, debugging용으로 token-level 원본도 보존한다.

지원할 XAI method:

- Integrated Gradients
- Occlusion
- 필요 시 Attention/Attention x Gradient

점수 방향을 통일한다.

### 8.1 Positive-direction score 형식

가능하면 XAI score를 positive logit 방향 기준 signed score로 저장한다.

```text
xai_score > 0: positive 방향 기여
xai_score < 0: negative 방향 기여
```

이 경우 예측을 지지하는 단어 점수는 다음처럼 계산한다.

```text
y_pred = +1 for positive prediction, -1 for negative prediction
support_score = y_pred * xai_score
```

`support_score`가 큰 단어일수록 현재 예측 감정을 지지하는 단어다.

### 8.2 Predicted-class score 형식

Occlusion처럼 predicted class probability/logit drop으로 계산한 경우:

```text
support_score = drop in predicted-class score after removing token
```

즉 값이 클수록 현재 예측을 지지하는 단어로 둔다.

## 9. 단어 제거 및 masking 실험

각 샘플마다 다음 변형을 만든다.

1. 원본 문장
2. XAI top-k supporting words 제거
3. XAI bottom-k words 제거
4. random k words 제거
5. XAI top-k supporting words `[MASK]` 치환
6. random k words `[MASK]` 치환

기본 `k`:

```text
k = 1, 2, 3
```

random control은 sample마다 20~50회 반복한다.

삭제 방식과 masking 방식은 둘 다 저장한다.

- 삭제: 실제 단어가 없어졌을 때 감정 표현이 약해지는지 보기 쉬움
- `[MASK]`: 문장 길이와 위치 구조를 더 잘 보존함

최종 주장에는 `[MASK]` 결과를 primary로 두고, deletion 결과를 robustness check로 둔다.

## 10. Direction-specific Movement 분석

각 변형 문장 `x'`에 대해 원본 `x`와 비교한다.

```text
h0 = h(x)
h1 = h(x')
delta_h = h1 - h0
```

계산 지표:

```text
projection_original = (h0 - center) dot v
projection_modified = (h1 - center) dot v
delta_projection = projection_modified - projection_original

y_pred = +1 if original prediction is positive else -1
strength_original = y_pred * projection_original
strength_modified = y_pred * projection_modified
delta_strength = strength_original - strength_modified

parallel_norm = abs(delta_h dot v)
orthogonal_norm = sqrt(max(||delta_h||^2 - parallel_norm^2, 0))
sentiment_ratio = parallel_norm / (||delta_h|| + eps)
alignment = cosine(delta_h, -y_pred * v)
```

중요한 판별 기준:

```text
delta_strength > 0
alignment > 0
sentiment_ratio가 random/bottom보다 큼
```

이 세 조건이 함께 나와야 "XAI 중요 단어 제거가 특히 sentiment direction을 움직였다"고 말할 수 있다.

## 11. 통계 검정

sample 단위 paired comparison을 사용한다.

비교:

```text
XAI top-k vs random-k
XAI top-k vs bottom-k
masking vs deletion
IG vs Occlusion
layer별 차이
```

주요 metric:

```text
delta_strength
alignment
sentiment_ratio
```

검정:

- paired t-test: 분포가 비교적 정규적일 때
- Wilcoxon signed-rank test: 기본 권장
- bootstrap 95% confidence interval
- correlation: removed XAI mass와 delta_strength의 Spearman correlation

보고할 값:

```text
mean
median
std
bootstrap_ci_low
bootstrap_ci_high
p_value
effect_size
```

## 12. 출력 파일 설계

추천 출력 디렉터리:

```text
XAI/outputs_sentiment_direction/
```

파일:

```text
direction_layer_XX.npz
projection_scores.csv
layer_projection_summary.csv
linear_probe_metrics.csv
xai_word_scores_eojeol.csv
xai_sentiment_direction_ablation.csv
xai_sentiment_direction_ablation_summary.csv
sentiment_direction_case_report.md
```

`xai_sentiment_direction_ablation.csv` 컬럼:

```text
sample_id
text
true_label
prediction
probability
layer
layer_role
method
k
edit_type
control_type
removed_words
removed_indices
xai_removed_mass
projection_original
projection_modified
delta_projection
strength_original
strength_modified
delta_strength
delta_hidden_norm
parallel_norm
orthogonal_norm
sentiment_ratio
alignment
modified_prediction
modified_probability
```

`control_type` 값:

```text
xai_top
xai_bottom
random
```

`edit_type` 값:

```text
mask
delete
```

## 13. 시각화

보고서용 figure:

1. Layer별 projection ROC-AUC line plot
2. Positive/negative projection score histogram
3. Probe 성능 vs projection 성능 비교
4. XAI top-k/random/bottom의 `delta_strength` box plot
5. XAI top-k/random/bottom의 `alignment` box plot
6. XAI removed mass vs `delta_strength` scatter plot
7. case별 원본/제거 문장 projection 이동 화살표

case report 예시:

```text
case_001
original: 배우 연기는 좋은데 각본이 너무 별로다
prediction: negative
top XAI words: 각본, 별로다

projection_original: -2.41
projection_masked: -0.83
delta_strength: 1.58
alignment: 0.72
sentiment_ratio: 0.61

interpretation:
XAI가 고른 부정 근거 단어를 mask했을 때 hidden state가 긍정/부정 축에서 중립 방향으로 이동했다.
```

## 14. 성공 기준

다음 조건을 만족하면 강한 결과로 본다.

1. `v_sentiment` projection이 validation/test에서 label과 logit margin을 잘 설명한다.
2. Linear Probe가 좋은 성능을 보이고, probe weight가 `v_sentiment`와 양의 cosine similarity를 가진다.
3. XAI top-k 제거의 `delta_strength`가 random/bottom 제거보다 유의미하게 크다.
4. XAI top-k 제거의 `alignment`가 양수이고 random/bottom보다 크다.
5. XAI top-k 제거의 `sentiment_ratio`가 random/bottom보다 크다.
6. removed XAI mass와 `delta_strength` 사이에 양의 상관이 있다.
7. 위 결과가 마지막 layer뿐 아니라 primary_candidate layer에서도 유지된다.

이 조건들이 충족되면 다음처럼 주장할 수 있다.

```text
Transformer의 CLS hidden state에는 긍정/부정 감정 방향이 형성되어 있으며,
XAI가 중요하다고 판단한 단어는 단순히 output score에만 영향을 주는 것이 아니라
hidden state를 해당 감정 방향으로 밀어주는 역할을 한다.
```

단, 마지막 layer에서만 위 조건이 성립하면 다음처럼 제한해서 주장한다.

```text
마지막 layer의 CLS hidden state에서는 긍정/부정 감정 방향이 뚜렷하게 관찰되었으나,
이 결과는 classifier head 직전 표현에 대한 분석이므로 encoder 내부의 일반적 감정 representation으로 확대 해석하지 않는다.
```

## 15. 실패 또는 약한 결과의 해석

### Projection은 좋지만 XAI 제거 alignment가 약함

감정 방향은 존재하지만, 해당 XAI method가 고른 단어가 그 방향을 직접 움직이지 않을 수 있다. XAI method가 output local sensitivity를 잡는 반면, sentiment direction은 문장 전체 representation을 잡기 때문에 차이가 날 수 있다.

### Probe는 좋은데 Projection이 약함

감정 정보가 hidden state에 있지만 단순 평균 차이 하나로는 충분하지 않을 수 있다. 이 경우 class boundary가 더 복잡하거나 layer 선택이 잘못되었을 수 있다.

### XAI top-k와 random 차이가 작음

중요 단어 제거 방식이 너무 거칠거나, subword/eojeol alignment가 불안정하거나, 모델이 문장 전체 문맥에 의존하고 있을 수 있다.

### Deletion은 효과가 큰데 Masking은 약함

삭제가 문장 구조 자체를 크게 바꾸는 OOD perturbation일 수 있다. 이 경우 최종 결론에는 masking 결과를 더 중시한다.

## 16. 구현 단계

### Phase 1. Hidden state extraction

새 스크립트:

```text
XAI/Transformer/sentiment_direction.py
```

기능:

- model/tokenizer load
- NSMC train/validation/test read
- hidden states extraction
- layer별 CLS 저장
- `layer_role` 부여
- final layer와 primary_candidate layer 자동 식별
- model prediction/logit 저장

### Phase 2. Direction and projection metrics

기능:

- train split으로 `v_sentiment` 계산
- validation/test projection score 계산
- layer별 summary 생성
- final_reference 결과와 primary_candidate 결과를 분리해서 저장

### Phase 3. Linear probe

기능:

- layer별 logistic regression 학습
- validation/test 성능 저장
- probe weight와 `v_sentiment` cosine 저장

### Phase 4. Transformer XAI word scores

기능:

- selected sample에 대해 IG/Occlusion 실행
- subword score를 eojeol 단위로 집계
- raw score와 normalized score를 모두 저장

### Phase 5. XAI-based removal/masking

새 스크립트:

```text
XAI/Transformer/sentiment_direction_ablation.py
```

기능:

- XAI top/bottom/random words 선택
- delete/mask 변형 문장 생성
- 원본/변형 hidden state 추출
- `delta_strength`, `alignment`, `sentiment_ratio` 계산

### Phase 6. Report generation

기능:

- aggregate summary CSV 생성
- qualitative case report 생성
- plots 생성

## 17. 실행 예시

아래 명령은 구현 후 목표 형태다.

```powershell
python XAI/Transformer/sentiment_direction.py `
  --model-dir XAI/Transformer/kcelectra_nsmc_model `
  --train-file Data/NSMC/ratings_train.txt `
  --test-file Data/NSMC/ratings_test.txt `
  --output-dir XAI/outputs_sentiment_direction `
  --device cuda
```

```powershell
python XAI/Transformer/sentiment_direction_ablation.py `
  --model-dir XAI/Transformer/kcelectra_nsmc_model `
  --direction-dir XAI/outputs_sentiment_direction `
  --xai-scores XAI/outputs_sentiment_direction/xai_word_scores_eojeol.csv `
  --output-dir XAI/outputs_sentiment_direction `
  --top-k 1,2,3 `
  --random-repeats 30 `
  --device cuda
```

## 18. 보고서 문장 예시

강한 결과가 나왔을 때:

```text
본 실험에서는 KcELECTRA 감정분류 모델의 layer별 CLS hidden state에서 positive centroid와 negative centroid의 차이로 sentiment direction을 정의하였다. 마지막 layer는 classifier head에 가까우므로 reference로만 사용하고, primary 분석은 중간-후반 layer에서 수행하였다. 해당 방향의 projection score는 label 및 logit margin과 일관된 관계를 보였고, linear probe 결과도 CLS 표현에 감정 정보가 선형적으로 포함되어 있음을 지지하였다. 또한 XAI가 중요 단어로 선택한 표현을 mask했을 때 hidden state 변화는 random/bottom 단어 제거보다 더 크게 원래 감정을 약화시키는 방향으로 정렬되었다. 이는 XAI가 선택한 단어가 output score뿐 아니라 Transformer encoder 내부의 sentiment representation에도 실질적인 영향을 준다는 근거로 해석할 수 있다.
```

약한 결과가 나왔을 때:

```text
Sentiment direction 자체는 projection 및 probe 기준에서 확인되었지만, XAI 중요 단어 제거 후 hidden state 변화가 해당 direction에 강하게 정렬되지는 않았다. 이는 단어 단위 XAI attribution과 문장 단위 CLS representation이 서로 다른 수준의 설명을 제공함을 시사한다.
```

## 19. 최종 체크리스트

- [ ] `v_sentiment`는 train split으로만 계산했는가?
- [ ] validation/test leakage가 없는가?
- [ ] layer별 projection 성능을 비교했는가?
- [ ] final layer 결과를 classifier-proximal reference로 분리했는가?
- [ ] primary_candidate 중간-후반 layer에서도 핵심 결과가 유지되는가?
- [ ] 마지막 layer 결과만으로 encoder 내부 representation이라고 과장하지 않았는가?
- [ ] Linear Probe와 projection baseline을 둘 다 보고했는가?
- [ ] XAI score의 방향 의미를 명확히 정의했는가?
- [ ] subword와 eojeol alignment를 저장했는가?
- [ ] XAI top-k, bottom-k, random-k control을 모두 비교했는가?
- [ ] deletion과 masking 결과를 분리해서 보고했는가?
- [ ] `delta_strength`, `alignment`, `sentiment_ratio`를 모두 계산했는가?
- [ ] 통계 검정과 confidence interval을 포함했는가?
- [ ] qualitative case report에서 수치와 해석을 함께 제시했는가?
