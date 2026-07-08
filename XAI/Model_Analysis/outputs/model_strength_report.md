# 모델 강점 분석 리포트

이 리포트는 `XAI/outputs_json`에 저장된 예측 확률과 단어별 XAI score를 사용해 모델별 설명 특성을 비교합니다.
최종 모델 순위는 모든 모델에 공통으로 존재하는 XAI 방법만 사용해 계산합니다.
최종 모델 순위는 모든 모델에 공통으로 존재하는 입력 문장 **30개**만 사용해 계산합니다.
현재 모델 비교에 사용된 공통 방법: **Integrated Gradients 100 steps, Integrated Gradients 50 steps, Occlusion**.

주의: `outputs_json`에는 정답 라벨이 없으므로 이 분석은 분류 정확도 평가가 아닙니다. 대신 모델의 설명 score 형태를 비교합니다.

## 부문별 우수 모델

- Attribution Sign Agreement: **Transformer** (0.8620)
- Max Attribution Share: **CNN** (0.5074)
- Attribution Coverage: **FNN** (0.8607)
- Top-3 Attribution Mass: **CNN** (0.8156)
- Explanation Agreement: **FNN** (0.8648)

## 모델별 요약

- **CNN**: Attribution Sign Agreement=0.8609, Max Attribution Share=0.5074, Attribution Coverage=0.7860, Top-3 Attribution Mass=0.8156, Explanation Agreement=0.8015
- **FNN**: Attribution Sign Agreement=0.8190, Max Attribution Share=0.4216, Attribution Coverage=0.8607, Top-3 Attribution Mass=0.7760, Explanation Agreement=0.8648
- **Transformer**: Attribution Sign Agreement=0.8620, Max Attribution Share=0.4382, Attribution Coverage=0.8599, Top-3 Attribution Mass=0.7754, Explanation Agreement=0.5776

## 순위 산정 방법

- **Attribution Sign Agreement**: 예측 감성과 같은 방향의 attribution score가 전체 설명량에서 차지하는 비율입니다. 예측 방향과 같은 부호의 설명량 비율이 높은 모델을 우수 모델로 봅니다.
- **Max Attribution Share**: 가장 큰 절대 score를 가진 단어 하나가 전체 설명량에서 차지하는 비율입니다. 핵심 단어를 더 뚜렷하게 잡아내는 모델을 우수 모델로 봅니다.
- **Attribution Coverage**: 문장 내 단어 중 전체 설명량의 2% 이상을 차지한 단어의 비율입니다. 의미 있는 크기의 설명 신호를 더 많은 단어에 부여한 모델을 우수 모델로 봅니다.
- **Top-3 Attribution Mass**: 절대 score 기준 상위 3개 단어가 전체 설명량에서 차지하는 비율입니다. 중요 단어 3개에 설명이 선명하게 모이는 모델을 우수 모델로 봅니다.
- **Explanation Agreement**: 같은 모델 안에서 여러 XAI 방법이 비슷한 핵심 단어를 고르는 정도입니다. Top-3 단어 겹침과 절대 score 순위 상관이 높은 모델을 우수 모델로 봅니다.

## 계산 방식 요약

- 단어별 score는 모델과 XAI 방법마다 크기 범위가 다를 수 있으므로, 문장 단위에서 절대값 합을 기준으로 비율형 지표를 계산합니다.
- `Attribution Coverage`는 정규화된 절대 score가 0.02 이상인 단어만 유효 설명 단어로 계산합니다.
- `Top-3 Attribution Mass`와 `Top-3 Jaccard Overlap`은 단어 수가 4개 이상인 문장에 대해서만 계산합니다.
- `Attribution Sign Agreement`는 positive 예측이면 양수 score, negative 예측이면 음수 score를 예측을 지지하는 설명으로 봅니다.
- `Explanation Agreement`는 같은 문장에 대해 방법별 상위 3개 단어 겹침도와 절대 score 순위 상관을 평균낸 값입니다.
- FNN에만 있는 LIME처럼 특정 모델에만 존재하는 방법은 방법별 참고표에는 남기지만, 최종 모델 순위에는 포함하지 않습니다.
