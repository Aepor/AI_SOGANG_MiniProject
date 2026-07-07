# 모델별 XAI 방법 비교

이 문서는 각 모델 내부에서 어떤 XAI 방법이 어떤 특성을 보이는지 정리합니다.
`method_strength_summary.csv`를 사람이 읽기 쉽게 해석한 파일이며, 모델 간 최종 우수 모델 비교와는 목적이 다릅니다.

## 해석 기준

- 예측 방향-설명 정합도: 모델 예측 방향과 XAI score 부호가 얼마나 잘 맞는지 봅니다.
- 유효 단어 커버리지: 전체 설명량의 2% 이상을 받은 단어 비율입니다.
- 핵심 단어 집중도/상위 3개 단어 집중도: 설명이 소수 단어에 얼마나 모이는지 보는 특성 지표입니다.
- 평균 절대 score: 단어별 score 크기의 평균입니다. 방법마다 scale이 다를 수 있어 같은 모델 내부 참고값으로만 봅니다.

## CNN

### 돋보이는 XAI 방법

- 예측 방향-설명 정합도: **Occlusion** (0.9580)
- 유효 단어 커버리지: **Filter Activation** (0.8926)
- 핵심 단어 집중도: **Occlusion** (0.4802)
- 상위 3개 단어 집중도: **Occlusion** (0.8056)
- 평균 절대 score: **Filter Activation** (0.1446)

### 방법별 특징

#### Filter Activation

- 원본 방법: Filter Activation
- 사례 수: 30
- 예측 방향-설명 정합도: 0.8613
- 유효 단어 커버리지: 0.8926
- 핵심 단어 집중도: 0.3585
- 상위 3개 단어 집중도: 0.7316
- 평균 절대 score: 0.1446
- 특징: CNN 필터가 강하게 반응한 단어를 보는 모델 특화 설명입니다. 다른 모델과의 최종 비교에는 넣지 않습니다.

#### Integrated Gradients 100 steps

- 원본 방법: Integrated Gradients 100 steps
- 사례 수: 30
- 예측 방향-설명 정합도: 0.8337
- 유효 단어 커버리지: 0.8019
- 핵심 단어 집중도: 0.4466
- 상위 3개 단어 집중도: 0.7945
- 평균 절대 score: 0.1446
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Integrated Gradients 50 steps

- 원본 방법: Integrated Gradients 50 steps
- 사례 수: 30
- 예측 방향-설명 정합도: 0.8336
- 유효 단어 커버리지: 0.8019
- 핵심 단어 집중도: 0.4467
- 상위 3개 단어 집중도: 0.7944
- 평균 절대 score: 0.1446
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Occlusion

- 원본 방법: N-gram Occlusion, Unigram Occlusion
- 사례 수: 30
- 예측 방향-설명 정합도: 0.9580
- 유효 단어 커버리지: 0.8164
- 핵심 단어 집중도: 0.4802
- 상위 3개 단어 집중도: 0.8056
- 평균 절대 score: 0.1398
- 특징: Unigram Occlusion과 N-gram Occlusion을 같은 문장 기준으로 평균내 CNN의 Occlusion 계열 설명으로 묶었습니다.

## FNN

### 돋보이는 XAI 방법

- 예측 방향-설명 정합도: **Occlusion** (0.8652)
- 유효 단어 커버리지: **Integrated Gradients 50 steps** (0.8868)
- 핵심 단어 집중도: **Occlusion** (0.5072)
- 상위 3개 단어 집중도: **Occlusion** (0.8225)
- 평균 절대 score: **Integrated Gradients 50 steps** (0.1418)

### 방법별 특징

#### Integrated Gradients 100 steps

- 원본 방법: Integrated Gradients 100 steps
- 사례 수: 30
- 예측 방향-설명 정합도: 0.7958
- 유효 단어 커버리지: 0.8756
- 핵심 단어 집중도: 0.3785
- 상위 3개 단어 집중도: 0.7524
- 평균 절대 score: 0.1395
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Integrated Gradients 50 steps

- 원본 방법: Integrated Gradients 50 steps
- 사례 수: 30
- 예측 방향-설명 정합도: 0.7961
- 유효 단어 커버리지: 0.8868
- 핵심 단어 집중도: 0.3791
- 상위 3개 단어 집중도: 0.7531
- 평균 절대 score: 0.1418
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### LIME

- 원본 방법: LIME
- 사례 수: 30
- 예측 방향-설명 정합도: 0.7905
- 유효 단어 커버리지: 0.8547
- 핵심 단어 집중도: 0.3805
- 상위 3개 단어 집중도: 0.7607
- 평균 절대 score: 0.0924
- 특징: 입력 perturbation으로 만든 local surrogate 설명입니다. FNN 내부 참고용으로 유지합니다.

#### Occlusion

- 원본 방법: Occlusion
- 사례 수: 30
- 예측 방향-설명 정합도: 0.8652
- 유효 단어 커버리지: 0.8196
- 핵심 단어 집중도: 0.5072
- 상위 3개 단어 집중도: 0.8225
- 평균 절대 score: 0.0745
- 특징: 단어를 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다.

## Transformer

### 돋보이는 XAI 방법

- 예측 방향-설명 정합도: **Integrated Gradients 100 steps** (0.9259)
- 유효 단어 커버리지: **Attention** (0.9944)
- 핵심 단어 집중도: **Occlusion** (0.5341)
- 상위 3개 단어 집중도: **Occlusion** (0.8529)
- 평균 절대 score: **Integrated Gradients 100 steps** (0.1446)

### 방법별 특징

#### Attention

- 원본 방법: Attention
- 사례 수: 30
- 예측 방향-설명 정합도: 0.5000
- 유효 단어 커버리지: 0.9944
- 핵심 단어 집중도: 0.2501
- 상위 3개 단어 집중도: 0.5976
- 평균 절대 score: 0.1446
- 특징: 모델 내부 attention 가중치를 보는 참고형 설명입니다. score가 대부분 양수라 방향성 지표는 보조적으로만 해석합니다.

#### Integrated Gradients 100 steps

- 원본 방법: Integrated Gradients 100 steps
- 사례 수: 30
- 예측 방향-설명 정합도: 0.9259
- 유효 단어 커버리지: 0.8648
- 핵심 단어 집중도: 0.4164
- 상위 3개 단어 집중도: 0.7710
- 평균 절대 score: 0.1446
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Integrated Gradients 50 steps

- 원본 방법: Integrated Gradients 50 steps
- 사례 수: 30
- 예측 방향-설명 정합도: 0.8948
- 유효 단어 커버리지: 0.8543
- 핵심 단어 집중도: 0.4067
- 상위 3개 단어 집중도: 0.7703
- 평균 절대 score: 0.1446
- 특징: baseline에서 입력까지의 gradient 누적값으로 단어 기여도를 계산하는 gradient 기반 설명입니다.

#### Occlusion

- 원본 방법: Occlusion
- 사례 수: 30
- 예측 방향-설명 정합도: 0.8699
- 유효 단어 커버리지: 0.6989
- 핵심 단어 집중도: 0.5341
- 상위 3개 단어 집중도: 0.8529
- 평균 절대 score: 0.1446
- 특징: 단어를 가렸을 때 예측 확률이 얼마나 변하는지 보는 제거 기반 설명입니다.

