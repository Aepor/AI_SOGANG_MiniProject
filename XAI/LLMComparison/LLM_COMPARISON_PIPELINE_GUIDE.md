# LLM Comparison Pipeline Guide

이 문서는 `XAI/LLMComparison` 파이프라인이 현재 어떻게 동작하는지 설명한다. 핵심 목적은 CNN/FNN/Transformer XAI 결과가 OpenAI API로 수집한 LLM의 자연어 근거와 얼마나 겹치는지 정량/정성적으로 평가하는 것이다.

LLM의 판단은 정답 라벨이 아니다. 여기서는 사람이 읽기 쉬운 고성능 LLM 설명을 비교 기준(reference explanation)으로 두고, 작은 모델의 XAI 결과가 같은 어절과 같은 polarity 방향을 잡는지 확인한다.

## 1. 전체 실행 명령

기본 실행은 아래 한 줄이다.

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py
```

처음에는 API 비용과 응답 형식을 확인하기 위해 일부 샘플만 실행하는 것이 좋다.

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --max-samples 5
```

실제로 실행하지 않고 어떤 단계가 돌지 확인하려면 `--dry-run`을 사용한다.

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --max-samples 5 --dry-run
```

## 2. 환경 설정

OpenAI API 키와 모델 설정은 repo root의 `.env`에서 읽는다.

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5-pro
OPENAI_REASONING_EFFORT=high
```

`.env`는 `.gitignore`에 들어가 있으므로 git에 올라가지 않는다. 공유용 예시는 `.env.example`에 둔다.

현재 기본 모델은 성능 우선 설정인 `gpt-5.5-pro`이고, reasoning effort는 `high`다. 분석 샘플 수가 적기 때문에 비용보다 근거 품질을 우선한다.

## 3. 입력 데이터

### 3.1 분석 샘플

기본 샘플 파일:

```text
XAI/LLMComparison/comparison_outputs/selected_reviews.csv
```

필수 컬럼은 다음이다.

```csv
sample_id,text,true_label,sample_type,selection_reason
```

`sample_id`는 이후 모든 결과 파일을 연결하는 key다. `text`는 XAI JSON 결과와 매칭할 때도 사용된다.

### 3.2 XAI 결과 JSON

기본 입력 디렉터리:

```text
XAI/outputs_json/
```

현재 지원하는 파일은 다음이다.

CNN:

```text
output_cnn_unigram_occlusion.json
output_cnn_ngram_occlusion.json
output_cnn_filter_activation.json
output_cnn_integrated_gradients_steps50.json
output_cnn_integrated_gradients_steps100.json
```

FNN:

```text
output_fnn_occlusion.json
output_fnn_lime.json
output_fnn_ig_50.json
output_fnn_ig_100.json
```

Transformer:

```text
output_transformer_occlusion.json
output_transformer_ig_50.json
output_transformer_ig_100.json
```

`output_transformer_attention.json`은 이번 LLM comparison에서 제외한다. attention score는 다른 XAI score와 의미가 달라 같은 ranking 표에 섞지 않기 위해서다.

각 JSON item은 다음 스키마를 기대한다. FNN처럼 `scores` 대신 `ig_scores`, `lime_scores`, `occlusion_scores` 같은 method별 score key를 쓰는 경우도 통합 단계에서 자동으로 인식한다.

```json
{
  "text": "리뷰 문장",
  "prediction": "positive",
  "probability": 0.98,
  "words": ["어절", "리스트"],
  "scores": [0.1, -0.2]
}
```

`words`와 `scores` 길이는 반드시 같아야 한다. `scores`는 positive class 방향 기준으로 해석한다. 양수는 positive 기여, 음수는 negative 기여다.

## 4. 파이프라인 단계

전체 runner는 아래 다섯 단계를 순서대로 실행한다.

```text
unify -> prompts -> collect -> normalize -> compare
```

구현 파일:

```text
XAI/LLMComparison/run_llm_xai_experiment.py
```

### 4.1 Stage 1: XAI 결과 통합

실행 모듈:

```text
XAI/LLMComparison/unify_xai_outputs.py
```

출력:

```text
XAI/LLMComparison/comparison_outputs/xai_unified.jsonl
```

역할:

- CNN/FNN/Transformer별 JSON 파일을 하나의 JSONL로 합친다.
- `selected_reviews.csv`의 `text`와 XAI output의 `text`를 비교해 `sample_id`를 붙인다.
- CNN/Transformer처럼 원본 exporter가 이미 L1 정규화한 점수는 그대로 보존한다.
- FNN처럼 원본 score가 L1 정규화되어 있지 않은 경우에는 통합 단계에서 비교용으로 한 번만 L1 정규화한다.
- XAI output의 어절 표면형이 `selected_reviews.csv`의 공백 기준 어절과 다르지만 길이가 같으면, 비교용 `words`는 selected review 기준으로 맞추고 원본은 `source_words`에 남긴다.
- XAI output의 `text`가 `selected_reviews.csv`에 없으면 잘못된 `sample_id` 매핑을 막기 위해 해당 row를 건너뛴다.
- 통합 단계에서는 `source_score_l1_sum`과 `source_scores_l1_normalized`로 L1 여부를 기록한다.
- 구버전 raw output을 강제로 다시 정규화해야 할 때만 `--renormalize-scores`를 사용한다.
- `model`, `method`, `source_file`을 붙여 나중에 method별 비교가 가능하게 한다.

통합 row 예시:

```json
{
  "sample_id": "case_001",
  "model": "cnn",
  "method": "ngram_occlusion",
  "text": "이 영화 정말 재미있어요 추천합니다",
  "prediction": "positive",
  "probability": 0.99,
  "words": ["이", "영화", "정말", "재미있어요", "추천합니다"],
  "scores": [0.0, 0.02, 0.12, 0.55, 0.31],
  "score_target": "positive",
  "score_normalization": "word_l1_exported",
  "source_score_l1_sum": 1.0,
  "source_scores_l1_normalized": true
}
```

현재 기본 설정으로 30개 샘플과 12개 method를 통합하면 `30 x 12 = 360` rows가 생성된다.

### 4.2 Stage 2: LLM prompt 생성

실행 모듈:

```text
XAI/LLMComparison/llm_prompt_builder.py
```

출력:

```text
XAI/LLMComparison/comparison_outputs/llm_prompts.jsonl
```

역할:

- `selected_reviews.csv`를 읽는다.
- 각 문장을 `text.split()`으로 어절 리스트로 나눈다.
- LLM에게 원문과 어절 리스트만 보여준다.
- XAI 결과는 prompt에 넣지 않는다.

LLM에게 요구하는 출력은 다음 구조다.

```json
{
  "sample_id": "case_001",
  "text": "원문",
  "words": ["어절", "리스트"],
  "sentiment": "positive|negative",
  "evidence": [
    {
      "phrase": "근거 구절",
      "word_indices": [0, 1],
      "polarity": "positive|negative",
      "strength": 0.8,
      "reason": "근거 설명"
    }
  ],
  "brief_reason": "전체 판단 요약"
}
```

중요한 점은 `word_indices`다. LLM이 자유롭게 구절을 말하는 것이 아니라, 제공된 `words` 배열의 index로 근거 위치를 지정해야 XAI score와 비교할 수 있다.

### 4.3 Stage 3: LLM 응답 수집

실행 모듈:

```text
XAI/LLMComparison/collect_llm_explanations.py
```

출력:

```text
XAI/LLMComparison/comparison_outputs/llm_explanations.jsonl
```

기본 모드는 OpenAI API mode다.

```powershell
python XAI/LLMComparison/collect_llm_explanations.py --mode api
```

수동 JSONL을 쓰고 싶으면 manual mode를 사용한다.

```powershell
python XAI/LLMComparison/collect_llm_explanations.py --mode manual
```

수집 스크립트의 특징:

- `.env`에서 `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT`를 읽는다.
- OpenAI Responses API에 structured output JSON schema를 보낸다.
- 이미 `llm_explanations.jsonl`에 있는 `sample_id`는 기본적으로 다시 호출하지 않는다.
- `--overwrite`를 붙이면 기존 sample도 다시 호출한다.
- `--max-samples 5`처럼 일부만 호출할 수 있다.
- API 호출 후 한 row씩 즉시 저장하므로 중간에 멈춰도 이어서 실행하기 쉽다.

저장되는 row에는 LLM 출력 외에도 다음 metadata가 붙는다.

```text
model_id
prompt_version
created_at
api_provider
response_id
```

### 4.4 Stage 4: LLM evidence vector 정규화

실행 모듈:

```text
XAI/LLMComparison/normalize_llm_outputs.py
```

출력:

```text
XAI/LLMComparison/comparison_outputs/llm_vectors.jsonl
```

역할:

- LLM 응답 JSONL을 검증한다.
- `sentiment`가 `positive|negative`인지 확인한다.
- `word_indices`가 `0 <= idx < len(words)` 범위 안에 있는지 확인한다.
- evidence의 `polarity`와 `strength`를 이용해 어절 단위 signed vector를 만든다.

vector 생성 규칙:

```text
positive evidence: +strength
negative evidence: -strength
같은 어절에 여러 evidence가 겹치면 합산
마지막에 L1 정규화
```

예를 들어 LLM이 2번 어절을 negative evidence로 strength 0.8이라고 찍으면 해당 위치에는 음수 값이 들어간다. 이렇게 만들어진 `llm_vector`가 XAI의 `scores`와 같은 길이를 갖게 된다.

### 4.5 Stage 5: XAI와 LLM evidence 비교

실행 모듈:

```text
XAI/LLMComparison/compare_llm_with_xai.py
```

출력:

```text
XAI/LLMComparison/comparison_outputs/llm_xai_overlap_scores.csv
XAI/LLMComparison/comparison_outputs/llm_xai_method_summary.csv
XAI/LLMComparison/comparison_outputs/qualitative_case_report.md
XAI/LLMComparison/comparison_outputs/llm_xai_evaluation_report.md
```

역할:

- `xai_unified.jsonl`과 `llm_vectors.jsonl`을 `sample_id`로 join한다.
- 각 `sample_id x model x method` 단위로 비교 지표를 계산한다.
- method별 평균 summary를 만든다.
- 정성 케이스 보고서와 최종 평가 보고서를 쓴다.

## 5. 비교 지표 의미

### 5.1 Top-k Recall

LLM이 evidence로 찍은 어절 중 XAI top-k 어절에 포함된 비율이다.

```text
topk_recall = |LLM evidence indices ∩ XAI top-k indices| / |LLM evidence indices|
```

기본 `k`는 3이다.

### 5.2 Jaccard

LLM evidence set과 XAI top-k set의 집합 겹침이다.

```text
jaccard = |A ∩ B| / |A ∪ B|
```

### 5.3 Signed Cosine

LLM evidence vector와 XAI score vector의 방향 유사도다. 어절 위치뿐 아니라 positive/negative polarity 방향도 본다.

```text
signed_cosine = cosine(llm_vector, xai_scores)
```

해석:

- 1에 가까움: 같은 어절을 같은 감정 방향으로 본다.
- 0 근처: 근거 위치나 방향이 거의 무관하다.
- 음수: 같은 어절을 반대 polarity로 해석했을 가능성이 있다.

### 5.4 Evidence Mass

LLM evidence 위치에 XAI 중요도가 얼마나 몰렸는지 본다.

```text
evidence_mass = sum(abs(xai_score[i]) for i in llm_evidence_indices)
```

XAI score는 L1 정규화되어 있으므로 값이 클수록 XAI가 LLM 근거 위치에 많은 중요도를 배정한 것이다.

### 5.5 Polarity Agreement

LLM evidence와 XAI top-k가 겹친 어절에서 부호가 같은 비율이다.

```text
polarity_agreement = same_sign_count / overlapping_evidence_count
```

### 5.6 LLM Match Score

발표와 method ranking을 위한 종합 점수다.

```text
llm_match_score =
  0.35 * topk_recall
+ 0.25 * jaccard
+ 0.25 * max(0, signed_cosine)
+ 0.15 * evidence_mass
```

이 점수는 편의상 만든 요약 지표다. 정답률처럼 해석하면 안 되고, 반드시 개별 지표와 함께 해석해야 한다.

### 5.7 Prediction Agreement

`prediction_llm_agreement`는 작은 모델의 prediction과 LLM sentiment가 같은지 나타낸다. 설명 overlap과는 분리해서 본다.

예를 들어 prediction은 LLM과 같지만 evidence가 다를 수 있고, prediction은 다르지만 일부 evidence 위치는 겹칠 수 있다.

## 6. 최종 산출물 읽는 법

### 6.1 `llm_xai_overlap_scores.csv`

가장 세밀한 결과 파일이다. 한 row는 하나의 `sample_id x model x method` 비교다.

주요 컬럼:

```text
sample_id
model
method
prediction
llm_sentiment
prediction_llm_agreement
topk_recall
jaccard
signed_cosine
evidence_mass
polarity_agreement
llm_match_score
xai_top_words
llm_evidence_words
```

디버깅이나 특정 case 분석은 이 파일에서 시작하면 된다.

### 6.2 `llm_xai_method_summary.csv`

method별 평균 성능표다.

한 row는 하나의 `model x method`다. 발표에서 어떤 XAI method가 LLM 근거와 가장 잘 맞는지 보여줄 때 사용한다.

### 6.3 `qualitative_case_report.md`

sample별로 LLM evidence와 각 XAI method의 top words를 나열한 케이스 리포트다. 케이스별 정성 분석을 빠르게 볼 때 사용한다.

### 6.4 `llm_xai_evaluation_report.md`

핵심 보고서다.

포함 내용:

- 실험 scope
- metric guide
- 전체 method ranking
- model-level summary
- best aligned rows
- weak/divergent rows
- sample별 best/weakest XAI 요약

앞으로 실험을 실행하면 이 파일까지 자동 생성된다.

## 7. 부분 실행 방법

전체 runner에는 skip 옵션이 있다.

LLM 호출은 이미 끝났고 비교만 다시 하고 싶을 때:

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --skip-unify --skip-prompts --skip-collect --skip-normalize
```

LLM 결과는 그대로 두고 vector 정규화와 비교만 다시 하고 싶을 때:

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --skip-unify --skip-prompts --skip-collect
```

XAI JSON이 바뀌어서 통합부터 다시 하되 LLM API 호출은 생략하고 싶을 때:

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --skip-collect
```

기존 LLM 응답을 다시 호출하고 싶을 때:

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --overwrite-llm
```

## 8. 주의사항

1. `selected_reviews.csv`의 `text`와 XAI output의 `text`가 달라지면 `sample_id` 매칭이 흔들릴 수 있다.
2. `words`와 `scores` 길이가 다르면 비교가 중단된다.
3. LLM prompt는 XAI 결과를 보지 않는다. 그래야 LLM evidence가 독립적인 reference가 된다.
4. LLM evidence는 정답이 아니라 비교 기준이다.
5. `llm_match_score`는 method ranking용 보조 지표이며, signed cosine/evidence mass/top-k recall을 함께 봐야 한다.
6. `.env`에는 실제 API key가 들어가므로 절대 commit하지 않는다.
7. `--max-samples`를 사용하면 기존 결과에 없는 sample만 추가 호출한다. 이미 수집된 sample은 기본적으로 건너뛴다.

## 9. 현재 기본 경로 요약

```text
XAI/outputs_json/
  output_cnn_*.json
  output_fnn_*.json
  output_transformer_*.json
  # output_transformer_attention.json은 제외

XAI/LLMComparison/comparison_outputs/
  selected_reviews.csv
  xai_unified.jsonl
  llm_prompts.jsonl
  llm_explanations.jsonl
  llm_vectors.jsonl
  llm_xai_overlap_scores.csv
  llm_xai_method_summary.csv
  qualitative_case_report.md
  llm_xai_evaluation_report.md
```
