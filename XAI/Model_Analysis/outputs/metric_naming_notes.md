# XAI Metric Naming Notes

이 문서는 Model Analysis에서 쓰는 지표명이 기존 XAI 문헌의 표준 지표명인지,
또는 프로젝트 비교 목적에 맞게 만든 커스텀 지표인지 구분합니다.

## Literature-Aligned Names Used

| Current Name | Why This Name Is Used | Standardness |
|---|---|---|
| Top-3 Jaccard Overlap | Top-k 집합 유사도를 Jaccard overlap으로 계산합니다. Jaccard는 표준 유사도 지표입니다. | Standard component |
| Spearman Rank Correlation (Absolute Scores) | 절대 attribution score 순위의 상관을 Spearman correlation으로 계산합니다. Spearman은 표준 순위 상관 지표입니다. | Standard component |
| Top-3 Attribution Mass | 상위 k개 feature가 전체 attribution mass에서 차지하는 비율입니다. top-k attribution/rationale 분석에서 흔한 표현입니다. | Literature-aligned, project-specific k=3 |

## Project-Specific Metrics

| Current Name | Computation | Why It Is Project-Specific |
|---|---|---|
| Attribution Sign Agreement | 예측이 positive이면 양수 attribution, negative이면 음수 attribution이 전체 절대 attribution에서 차지하는 비율입니다. | signed attribution을 예측 감성 방향과 맞춘 프로젝트 지표입니다. 기존 논문에서 이 이름으로 고정된 표준 지표는 아닙니다. |
| Attribution Coverage | 전체 attribution mass의 2% 이상을 받은 단어의 비율입니다. | coverage/sparsity 계열 아이디어는 흔하지만 2% threshold와 단어 비율 계산은 프로젝트 설정입니다. |
| Max Attribution Share | 가장 큰 단일 단어 attribution mass가 전체 attribution mass에서 차지하는 비율입니다. | sparsity/concentration 아이디어를 단어 하나 기준으로 단순화한 프로젝트 지표입니다. |
| Top-3 Attribution Mass | 절대 score 기준 상위 3개 단어의 attribution mass 비율입니다. | top-k attribution mass라는 표현은 문헌 친화적이지만 k=3과 단어 단위 계산은 프로젝트 설정입니다. |
| Explanation Agreement | Top-3 Jaccard Overlap과 Spearman Rank Correlation을 평균낸 값입니다. | 구성요소는 표준적이지만 둘을 단순 평균해 하나의 점수로 만든 것은 프로젝트 지표입니다. |

## More Standard Alternatives Not Used Here

- Comprehensiveness: 중요한 rationale을 제거했을 때 예측이 얼마나 변하는지 보는 ERASER 계열 지표입니다.
- Sufficiency: rationale만 남겼을 때 예측이 얼마나 유지되는지 보는 ERASER 계열 지표입니다.
- Infidelity: explanation과 입력 perturbation에 따른 출력 변화가 얼마나 맞는지 보는 지표입니다.
- Sensitivity: 작은 입력 변화에 explanation이 얼마나 안정적인지 보는 지표입니다.
- ROAR: 중요한 feature를 제거하고 재학습해 feature importance를 평가하는 benchmark 방식입니다.

현재 프로젝트는 저장된 JSON의 단어별 score만 사용하므로, 재추론/제거 실험이 필요한 Comprehensiveness, Sufficiency, ROAR는 최종 지표로 쓰지 않았습니다.
