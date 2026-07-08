# 모델별 XAI 방법 비교

이 문서는 각 모델 내부에서 어떤 XAI 방법이 어떤 특성을 보이는지 정리합니다.
`method_strength_summary.csv`를 사람이 읽기 쉽게 해석한 파일이며, 모델 간 최종 특성 비교와는 목적이 다릅니다.

## 해석 기준

- Attribution Sign Agreement: 모델 예측 방향과 XAI score 부호가 얼마나 잘 맞는지 봅니다.
- Attribution Coverage: 전체 설명량의 2% 이상을 받은 단어 비율입니다.
- Top-3 Attribution Mass: 설명이 주요 단어 묶음에 얼마나 모이는지 보는 특성 지표입니다.

## CNN

### 돋보이는 XAI 방법

- Attribution Sign Agreement: **N-gram Occlusion** (0.9638)
- Attribution Coverage: **Filter Activation** (0.9032)
- Top-3 Attribution Mass: **Unigram Occlusion** (0.8525)

### 방법별 특징

#### Filter Activation

- 원본 방법: Filter Activation
- 사례 수: 30
- Attribution Sign Agreement: 0.8519
- Attribution Coverage: 0.9032
- Top-3 Attribution Mass: 0.7217
- 특징: CNN 필터가 강하게 반응한 단어를 보는 모델 특화 설명입니다. 다른 모델과의 최종 비교에는 넣지 않습니다.

#### Integrated Gradients 100 steps

- 원본 방법: Integrated Gradients 100 steps
- 사례 수: 30
- Attribution Sign Agreement: 0.8350
- Attribution Coverage: 0.8187
- Top-3 Attribution Mass: 0.7973
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Integrated Gradients 50 steps

- 원본 방법: Integrated Gradients 50 steps
- 사례 수: 30
- Attribution Sign Agreement: 0.8349
- Attribution Coverage: 0.8187
- Top-3 Attribution Mass: 0.7972
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### N-gram Occlusion

- 원본 방법: N-gram Occlusion
- 사례 수: 30
- Attribution Sign Agreement: 0.9638
- Attribution Coverage: 0.8345
- Top-3 Attribution Mass: 0.7624
- 특징: 연속된 여러 단어 묶음을 가렸을 때 예측 확률 변화를 보는 제거 기반 설명입니다. CNN 내부 방법 비교용으로 유지합니다.

#### Unigram Occlusion

- 원본 방법: Unigram Occlusion
- 사례 수: 30
- Attribution Sign Agreement: 0.9129
- Attribution Coverage: 0.7207
- Top-3 Attribution Mass: 0.8525
- 특징: 단어를 하나씩 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다. 모델 간 최종 비교에서 CNN의 Occlusion 대표값으로 사용합니다.

## FNN

### 돋보이는 XAI 방법

- Attribution Sign Agreement: **Occlusion** (0.8652)
- Attribution Coverage: **Integrated Gradients 50 steps** (0.8868)
- Top-3 Attribution Mass: **Occlusion** (0.8225)

### 방법별 특징

#### Integrated Gradients 100 steps

- 원본 방법: Integrated Gradients 100 steps
- 사례 수: 30
- Attribution Sign Agreement: 0.7958
- Attribution Coverage: 0.8756
- Top-3 Attribution Mass: 0.7524
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Integrated Gradients 50 steps

- 원본 방법: Integrated Gradients 50 steps
- 사례 수: 30
- Attribution Sign Agreement: 0.7961
- Attribution Coverage: 0.8868
- Top-3 Attribution Mass: 0.7531
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### LIME

- 원본 방법: LIME
- 사례 수: 30
- Attribution Sign Agreement: 0.7905
- Attribution Coverage: 0.8547
- Top-3 Attribution Mass: 0.7607
- 특징: 입력 perturbation으로 만든 local surrogate 설명입니다. FNN 내부 참고용으로 유지합니다.

#### Occlusion

- 원본 방법: Occlusion
- 사례 수: 30
- Attribution Sign Agreement: 0.8652
- Attribution Coverage: 0.8196
- Top-3 Attribution Mass: 0.8225
- 특징: 단어를 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다.

## Transformer

### 돋보이는 XAI 방법

- Attribution Sign Agreement: **Occlusion** (0.9455)
- Attribution Coverage: **Attention** (1.0000)
- Top-3 Attribution Mass: **Occlusion** (0.8426)

### 방법별 특징

#### Attention

- 원본 방법: Attention
- 사례 수: 30
- Attribution Sign Agreement: 0.4000
- Attribution Coverage: 1.0000
- Top-3 Attribution Mass: 0.6038
- 특징: 모델 내부 attention 가중치를 보는 참고형 설명입니다. score가 대부분 양수라 방향성 지표는 보조적으로만 해석합니다.

#### Integrated Gradients 100 steps

- 원본 방법: Integrated Gradients 100 steps
- 사례 수: 30
- Attribution Sign Agreement: 0.8441
- Attribution Coverage: 0.9042
- Top-3 Attribution Mass: 0.7430
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Integrated Gradients 50 steps

- 원본 방법: Integrated Gradients 50 steps
- 사례 수: 30
- Attribution Sign Agreement: 0.7963
- Attribution Coverage: 0.8866
- Top-3 Attribution Mass: 0.7406
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Occlusion

- 원본 방법: Occlusion
- 사례 수: 30
- Attribution Sign Agreement: 0.9455
- Attribution Coverage: 0.7888
- Top-3 Attribution Mass: 0.8426
- 특징: 단어를 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다.

