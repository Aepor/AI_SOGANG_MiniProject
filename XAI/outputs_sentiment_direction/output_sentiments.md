# 감정 방향성 XAI 분석 결과 파일 설명서 (Output Sentiments)

이 폴더(`outputs_sentiment_direction`)는 트랜스포머 모델의 내부 은닉 상태(Hidden State)를 분석하여 긍정/부정 감정 방향성을 도출하고, XAI 단어 제거 시 감정축의 변화를 확인하는 파이프라인의 최종 결과물들이 저장되는 곳입니다.

## 1차 구동 결과물 (감정 방향 기준축 도출)

15만 개의 학습 데이터를 기반으로 모델 내부의 '감정 축'을 세운 결과입니다.

*   **`direction_layer_00.npz` ~ `12.npz` (총 13개)**
    *   0층(임베딩)부터 12층까지 각 레이어별 **감정 방향 기준축(벡터) 원본 데이터** 행렬 파일입니다.
*   **`linear_probe_weights_layer_00.npz` ~ `12.npz` (총 13개)**
    *   각 층에서 학습된 선형 회귀(Linear Probe) 모델의 가중치 행렬 파일입니다.
*   **`projection_scores.csv`**
    *   전체 문장들이 계산된 각 층의 감정축 위에서 몇 점을 기록했는지 투영 점수(Projection Score)를 상세히 담은 파일입니다.
*   **`layer_projection_summary.csv`**
    *   각 층별 감정 분류 정확도(AUC) 및 투영 성능을 집계한 요약 표입니다.
*   **`linear_probe_metrics.csv`**
    *   각 층별 선형 회귀 모델의 예측 정확도(AUC, F1 등) 요약 표입니다.

## 2차 구동 결과물 (XAI 어블레이션 및 통계 분석)

설정된 감정 기준축을 바탕으로 30개 샘플 문장의 XAI 중요 단어를 제거/마스킹했을 때 일어나는 내부 수치 변화를 측정한 결과입니다.

*   **`xai_word_scores_eojeol.csv`**
    *   30개 문장에 대해 산출된 XAI(IG 기준)의 어절 단위 중요도(긍정/부정 지지도) 원본 점수입니다.
*   **`xai_sentiment_direction_ablation.csv`**
    *   중요 단어를 지우거나 가렸을 때, 모델 내부의 감정 수치가 몇에서 몇으로 얼마나 무너졌는지(`delta_strength`) 낱낱이 기록한 상세 실험 로그입니다.
*   **`xai_sentiment_direction_ablation_summary.csv`**
    *   **핵심 최종 통계 요약표**입니다. XAI가 고른 핵심 단어를 지웠을 때가, 아무 단어나 무작위(Random)로 지웠을 때보다 감정을 훨씬 유의미하게 깎아내렸는지 검증한 Wilcoxon 통계 검정 P-value 수치가 포함되어 있습니다.
