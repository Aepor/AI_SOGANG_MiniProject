# XAI 코드 분리 및 Opus 기준 비교 확장 계획

## 1. 개선 목표

현재 `XAI/CNN/nsmc_cnn_xai.py`는 CNN XAI 분석을 한 번에 실행할 수 있다는 장점이 있지만, 코드 이해와 발표 준비 관점에서는 다음 문제가 있다.

```text
1. 데이터 준비, 모델 로드, 샘플 선정, XAI 기법, 시각화, 요약 생성이 한 파일에 모여 있다.
2. Unigram Occlusion, n-gram Occlusion, Filter Activation, Max-pooling, Gradient x Input의 코드 경계가 분명하지 않다.
3. FNN/CNN/Transformer XAI 결과를 같은 형식으로 비교하기 위한 공통 스키마가 아직 없다.
4. Opus 기준 분석을 추가하려면 LLM 응답 저장 형식, 프롬프트 버전, 비교 지표를 별도로 정의해야 한다.
```

따라서 개선 목표는 다음과 같다.

```text
1. 각 XAI 방법을 독립 파일 또는 독립 notebook으로 분리한다.
2. 공통 전처리, 샘플 선정, 결과 저장, 시각화 코드는 shared 모듈로 묶는다.
3. CNN뿐 아니라 FNN, Transformer도 같은 출력 형식으로 맞출 수 있게 한다.
4. Opus를 정답 모델이 아니라 고성능 LLM의 정성적 기준 설명으로 추가한다.
5. 작은 모델의 XAI 결과와 Opus 자연어 판단 근거가 얼마나 비슷한지 비교한다.
```

## 2. 중요한 해석 원칙

Opus는 정답 모델이 아니다.

이 프로젝트에서 Opus는 다음 역할로만 사용한다.

| 대상 | 역할 |
|---|---|
| FNN | 단어 중심 small model |
| CNN | 구절/n-gram 중심 small model |
| Transformer | 문맥 기반 fine-tuned model |
| Opus | 고성능 LLM의 자연어 판단 기준, qualitative baseline, teacher-style reference |

발표나 보고서에서는 반드시 다음 표현을 사용한다.

```text
Opus의 설명은 절대적 정답이 아니라, 사람이 이해하기 쉬운 고성능 LLM 기준 설명으로 사용하였다.
본 프로젝트에서는 이를 FNN, CNN, Transformer의 XAI 결과와 비교하여 모델별 판단 근거 차이를 정성적으로 분석한다.
```

또한 Opus에게 이유를 묻는 것은 사후 자연어 설명이다. 반면 XAI는 occlusion, gradient, attention 등 모델 입력/내부값 기반 분석이다. 두 결과가 비슷하다고 해서 작은 모델이 Opus와 같은 방식으로 생각했다고 말하면 안 된다.

권장 표현:

```text
Transformer의 XAI 결과는 Opus가 자연어로 제시한 판단 근거와 가장 높은 표현 겹침을 보였다.
```

피해야 할 표현:

```text
Transformer는 Opus와 같은 이유로 판단했다.
Opus가 정답이므로 CNN 설명은 틀렸다.
```

## 3. 목표 폴더 구조

현재 구조를 크게 깨지 않으면서 `XAI` 아래에 공통 모듈과 방법별 모듈을 추가한다.

```text
XAI/
  shared/
    __init__.py
    paths.py
    nsmc_data.py
    tokenization.py
    samples.py
    schemas.py
    metrics.py
    visualization.py

  CNN/
    nsmc_cnn.ipynb
    best_cnn_model.pt
    cnn_preprocess_cache.pkl
    nsmc_cnn_xai.py                  # 기존 통합 실행 파일, 당분간 유지
    xai_methods/
      __init__.py
      model.py
      loader.py
      unigram_occlusion.py
      ngram_occlusion.py
      filter_activation.py
      maxpool_positions.py
      gradient_x_input.py
      integrated_gradients.py
      run_all.py
    notebooks/
      00_cnn_xai_overview.ipynb
      01_cnn_unigram_occlusion.ipynb
      02_cnn_ngram_occlusion.ipynb
      03_cnn_filter_activation.ipynb
      04_cnn_maxpool_positions.ipynb
      05_cnn_gradient_x_input.ipynb
      06_cnn_integrated_gradients.ipynb
    xai_outputs/

  FNN/
    notebooks/
      01_fnn_occlusion.ipynb
      02_fnn_gradient_x_input.ipynb
    xai_outputs/

  Transformer/
    notebooks/
      01_transformer_occlusion.ipynb
      02_transformer_integrated_gradients.ipynb
      03_transformer_attention.ipynb
    xai_outputs/

  OpusBaseline/
    prompt_templates.md
    opus_prompt_builder.py
    collect_opus_explanations.py
    normalize_opus_outputs.py
    compare_opus_with_xai.py
    notebooks/
      01_opus_prompt_preview.ipynb
      02_opus_xai_comparison.ipynb
    outputs/

  comparison_outputs/
    selected_reviews_20_30.csv
    model_predictions_unified.csv
    xai_topk_unified.csv
    opus_explanations.jsonl
    opus_xai_overlap_scores.csv
    qualitative_case_report.md
```

핵심 방향은 다음과 같다.

```text
스크립트:
반복 실행, CSV 생성, 대량 샘플 분석용

notebook:
방법별 이해, 그림 확인, 발표용 사례 해석용
```

즉, 계산은 `.py`에서 안정적으로 하고, 해석과 시각화는 `.ipynb`에서 빠르게 확인한다.

## 4. CNN 코드 분리 계획

현재 `nsmc_cnn_xai.py`에서 분리할 단위는 다음과 같다.

### 4.1 `XAI/CNN/shared/paths.py`

역할:

```text
1. repo root 탐색
2. XAI/CNN, XAI/Data/NSMC, output path 정리
3. Windows/PowerShell 기준 경로 안정화
```

포함 함수:

```python
repo_root()
xai_root()
find_data_file(filename)
ensure_dir(path)
```

### 4.2 `XAI/CNN/shared/nsmc_data.py`

역할:

```text
1. ratings_train.txt, ratings_test.txt 로드
2. 중복/결측 제거
3. train/validation split 재현
4. test sample pool 생성
```

포함 함수:

```python
load_nsmc_raw()
clean_nsmc_frame(df)
split_train_validation(train_df, validation_ratio=0.1, seed=42)
sample_test_pool(test_df, pool_size, seed)
```

### 4.3 `XAI/CNN/shared/tokenization.py`

역할:

```text
1. Okt 초기화
2. 노트북과 동일한 stopword 제거
3. token -> id encoding
4. pad/unk 처리
```

포함 함수:

```python
make_okt()
tokenize_text(text)
encode_tokens(tokens, word_to_index, max_len)
decode_ids(ids, index_to_word)
```

### 4.4 `XAI/CNN/shared/schemas.py`

역할:

```text
1. 모델별 예측 결과 형식 통일
2. XAI top-k 결과 형식 통일
3. Opus 결과 형식 통일
```

공통 CSV schema:

```text
sample_id
source
text
true_label
model_name
pred_label
target_class
confidence
method
rank
evidence_type
evidence_text
start_pos
end_pos
score
score_type
notes
```

이 형식이 중요하다. FNN, CNN, Transformer, Opus를 비교하려면 XAI 결과를 모두 `evidence_text` 중심으로 맞춰야 한다.

### 4.5 `XAI/CNN/xai_methods/model.py`

역할:

```text
1. CNN_Sentiment 클래스만 보관
2. 노트북 모델 구조와 1:1 대응
3. forward_from_embedded_for_xai 같은 gradient용 forward 포함
```

포함 항목:

```python
class CNN_Sentiment(nn.Module)
forward_from_embedded_for_xai(model, embedded)
```

### 4.6 `XAI/CNN/xai_methods/loader.py`

역할:

```text
1. best_cnn_model.pt 로드
2. state_dict에서 architecture 추론
3. vocab cache 로드/재생성
4. vocab size와 embedding size 검증
```

포함 함수:

```python
build_or_load_vocab_cache(cache_path, refresh_cache=False)
infer_architecture_from_state_dict(state_dict)
load_cnn_model(model_path, vocab_size, device)
```

### 4.7 `XAI/CNN/xai_methods/unigram_occlusion.py`

역할:

```text
token 하나를 <pad>로 치환했을 때 target class 확률이 얼마나 감소하는지 계산한다.
```

입력:

```text
model
sample_records
pad_idx
target_class
```

출력:

```text
cnn_unigram_occlusion.csv
unified xai_topk rows
```

주요 함수:

```python
run_unigram_occlusion(model, samples, pad_idx, device, batch_size)
extract_topk_unigram(rows, top_k=5)
```

### 4.8 `XAI/CNN/xai_methods/ngram_occlusion.py`

역할:

```text
n=1,2,3,4,5 연속 token window를 <pad>로 치환한다.
CNN filter size 3,4,5와 직접 비교하기 위한 핵심 방법이다.
```

주요 함수:

```python
run_ngram_occlusion(model, samples, pad_idx, ngram_sizes, device, batch_size)
extract_topk_ngram(rows, top_k=5)
```

### 4.9 `XAI/CNN/xai_methods/filter_activation.py`

역할:

```text
convolution filter가 어떤 n-gram에 강하게 반응하는지 추출한다.
fc weight 방향성을 함께 계산하여 긍정/부정 방향 filter를 구분한다.
```

주요 함수:

```python
get_conv_activations(model, ids, device)
run_filter_activation_analysis(model, samples, device)
compute_filter_class_direction(model)
```

### 4.10 `XAI/CNN/xai_methods/maxpool_positions.py`

역할:

```text
각 filter가 max-pooling으로 선택한 위치와 n-gram을 추출한다.
target class contribution 기준 top-k evidence를 만든다.
```

주요 함수:

```python
run_maxpool_position_analysis(model, samples, device)
extract_topk_maxpool(rows, top_k=5)
```

### 4.11 `XAI/CNN/xai_methods/gradient_x_input.py`

역할:

```text
embedding 기준 Gradient x Input token attribution을 계산한다.
occlusion 결과와 비교하는 보조 지표로 사용한다.
```

주요 함수:

```python
run_gradient_x_input(model, samples, device)
extract_topk_gradient(rows, top_k=5)
```

### 4.12 `XAI/CNN/xai_methods/integrated_gradients.py`

역할:

```text
<pad> baseline에서 실제 입력 embedding까지의 경로를 따라 target logit gradient를 누적한다.
Gradient x Input보다 안정적인 token-level attribution 보조 지표로 사용한다.
```

주요 함수:

```python
run_integrated_gradients(model, samples, pad_idx, device, steps=32)
extract_topk_integrated_gradients(rows, top_k=5)
```

### 4.13 `XAI/CNN/xai_methods/run_all.py`

역할:

```text
분리된 방법들을 한 번에 실행하는 CLI entrypoint.
기존 nsmc_cnn_xai.py의 사용자 경험은 유지한다.
```

실행 예:

```powershell
python XAI\CNN\xai_methods\run_all.py --sample-size 300 --pool-size 1000 --output-dir XAI\CNN\xai_outputs
```

기존 `nsmc_cnn_xai.py`는 당분간 compatibility wrapper로 둔다.

```python
from XAI.CNN.xai_methods.run_all import main

if __name__ == "__main__":
    raise SystemExit(main())
```

이렇게 하면 기존 명령을 쓰던 사람도 깨지지 않는다.

## 5. ipynb 분리 계획

notebook은 계산 전체를 다시 구현하는 곳이 아니라, 이미 생성된 CSV를 읽어서 방법별로 이해하고 해석하는 곳으로 둔다.

### 5.1 `00_cnn_xai_overview.ipynb`

목적:

```text
1. 현재 CNN 모델 구조 요약
2. metrics.json 확인
3. xai_outputs 파일 목록 확인
4. 대표 샘플 3개 미리보기
```

포함 셀:

```text
1. import pandas/pathlib
2. metadata json 로드
3. selected samples 로드
4. TP/TN/FP/FN 분포 출력
5. 대표 case summary 일부 출력
```

### 5.2 `01_cnn_unigram_occlusion.ipynb`

목적:

```text
token 하나씩 가렸을 때 모델 예측이 어떻게 바뀌는지 빠르게 이해한다.
```

포함 셀:

```text
1. cnn_unigram_occlusion.csv 로드
2. sample_id 선택 widget
3. token별 prob_drop bar chart
4. positive/negative 방향 해석 문장 생성
```

### 5.3 `02_cnn_ngram_occlusion.ipynb`

목적:

```text
CNN이 단어보다 n-gram에 얼마나 민감한지 확인한다.
```

포함 셀:

```text
1. cnn_ngram_occlusion.csv 로드
2. n=1,2,3,4,5별 top evidence 비교
3. filter size 3/4/5와 n-gram occlusion 연결
4. 발표용 top-k 표 생성
```

### 5.4 `03_cnn_filter_activation.ipynb`

목적:

```text
각 filter가 어떤 감정 n-gram에 반응하는지 확인한다.
```

포함 셀:

```text
1. cnn_filter_top_ngrams.csv 로드
2. filter_size, filter_idx 선택
3. 해당 filter의 top activation n-gram 보기
4. class direction 확인
5. 긍정/부정 filter 예시 표 생성
```

### 5.5 `04_cnn_maxpool_positions.ipynb`

목적:

```text
각 filter가 문장 내 어떤 위치를 최종 feature로 선택했는지 확인한다.
```

포함 셀:

```text
1. cnn_maxpool_positions.csv 로드
2. sample별 target_contribution top-k 출력
3. selected_ngram을 원문 token sequence 위에 강조
4. 오분류 사례 해석
```

### 5.6 `05_cnn_gradient_x_input.ipynb`

목적:

```text
gradient attribution을 occlusion 결과와 비교한다.
```

포함 셀:

```text
1. cnn_gradient_x_input.csv 로드
2. token별 normalized_abs_score heatmap
3. unigram occlusion top-k와 비교
4. 두 방법이 일치/불일치하는 사례 분석
```

### 5.7 `06_cnn_integrated_gradients.ipynb`

목적:

```text
Integrated Gradients token attribution을 확인하고 Gradient x Input과 비교한다.
```

포함 셀:

```text
1. cnn_integrated_gradients.csv 로드
2. sample별 IG top-k token 출력
3. Gradient x Input과 IG normalized score 비교
4. baseline=<pad>, ig_steps 해석 확인
```

## 6. FNN/CNN/Transformer 통합 비교 스키마

Opus와 비교하려면 세 모델의 XAI 결과를 하나의 형식으로 맞춰야 한다.

### 6.1 통합 예측 파일

파일:

```text
XAI/comparison_outputs/model_predictions_unified.csv
```

컬럼:

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 공통 샘플 id |
| `text` | 원문 리뷰 |
| `true_label` | NSMC label |
| `model_name` | `fnn`, `cnn`, `transformer` |
| `pred_label` | 모델 예측 |
| `negative_prob` | 부정 확률 |
| `positive_prob` | 긍정 확률 |
| `confidence` | 예측 class 확률 |
| `is_correct` | 정답 여부 |

### 6.2 통합 XAI top-k 파일

파일:

```text
XAI/comparison_outputs/xai_topk_unified.csv
```

컬럼:

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 공통 샘플 id |
| `model_name` | `fnn`, `cnn`, `transformer` |
| `method` | `occlusion`, `ngram_occlusion`, `gradient_x_input`, `integrated_gradients`, `attention` |
| `rank` | 중요도 순위 |
| `evidence_text` | 중요 단어/구절 |
| `evidence_tokens` | 비교용 token 문자열 |
| `score` | 중요도 점수 |
| `score_type` | `prob_drop`, `logit_drop`, `attribution`, `attention_weight` |
| `target_class` | 설명 대상 class |

### 6.3 모델별 evidence 해석 방식

| 모델 | 권장 evidence 단위 | 주 비교 기법 |
|---|---|---|
| FNN | 단어/token | Occlusion, Gradient x Input |
| CNN | n-gram phrase | n-gram Occlusion, Filter Activation, Max-pooling |
| Transformer | token/subword 또는 복원 phrase | Integrated Gradients, Occlusion, Attention 보조 |
| Opus | 자연어 핵심 표현 | LLM explanation |

모든 모델의 결과를 강제로 같은 의미라고 보지 않는다. 비교를 위해 `evidence_text`라는 공통 표면 형식으로 맞추되, 분석 단위의 차이는 보고서에서 설명한다.

## 7. Opus 기준 분석 추가 실험

### 7.1 실험 이름

```text
실험 7: Opus 기준 설명과 XAI 결과 비교
```

### 7.2 목적

```text
작은 딥러닝 모델의 XAI 결과가 고성능 LLM의 자연어 판단 근거와 얼마나 유사한지 정성적/정량적으로 비교한다.
```

핵심 질문:

```text
1. FNN/CNN/Transformer 중 어떤 모델의 중요 표현이 Opus의 설명과 가장 많이 겹치는가?
2. CNN의 n-gram evidence는 Opus가 제시한 핵심 표현과 잘 맞는가?
3. Transformer의 문맥 기반 XAI 결과는 반전 표현에서 Opus 설명과 더 잘 맞는가?
4. 오분류 사례에서 작은 모델이 Opus와 다르게 본 표현은 무엇인가?
```

### 7.3 샘플 선정

대표 문장 20~30개를 사용한다.

권장 구성:

| 유형 | 개수 | 목적 |
|---|---:|---|
| 명확한 긍정 | 5 | 쉬운 긍정 근거 비교 |
| 명확한 부정 | 5 | 쉬운 부정 근거 비교 |
| 긍정+부정 혼합 | 5 | 대조 표현 처리 비교 |
| 반전 표현 | 5 | `좋았지만`, `지루하지 않다` 같은 문맥 비교 |
| 모델 오분류 사례 | 5~10 | 모델별 약점 비교 |

샘플 파일:

```text
XAI/comparison_outputs/selected_reviews_20_30.csv
```

컬럼:

```text
sample_id
text
true_label
sample_type
selection_reason
```

### 7.4 Opus 프롬프트

기본 프롬프트:

```text
다음 한국어 영화 리뷰가 긍정인지 부정인지 판단하고,
그 판단의 근거가 되는 핵심 단어 또는 표현을 3개 이하로 뽑아줘.

리뷰:
"{review_text}"

출력 형식:
감정: 긍정/부정
핵심 근거 표현: [...]
간단한 이유: ...
```

자동 파싱을 쉽게 하려면 JSON 출력 프롬프트를 권장한다.

```text
다음 한국어 영화 리뷰가 긍정인지 부정인지 판단해줘.
판단 근거가 되는 핵심 단어 또는 표현은 최대 3개만 뽑아줘.

주의:
- 출력은 JSON만 작성해.
- 감정은 "positive" 또는 "negative" 중 하나로 작성해.
- 핵심 근거 표현은 원문에 실제로 등장하는 표현을 우선 사용해.
- 애매하면 더 강한 최종 평가 표현을 기준으로 판단해.
- 너의 설명은 정답 라벨이 아니라 정성적 비교 기준으로 사용된다.

리뷰:
"{review_text}"

JSON schema:
{
  "sentiment": "positive|negative",
  "evidence_phrases": ["...", "..."],
  "brief_reason": "..."
}
```

### 7.5 Opus 결과 저장 형식

파일:

```text
XAI/comparison_outputs/opus_explanations.jsonl
```

한 줄 예시:

```json
{"sample_id":"case_001","model_id":"claude-opus-*","prompt_version":"opus_sentiment_v1","text":"배우는 좋았지만 스토리는 너무 지루했다.","sentiment":"negative","evidence_phrases":["좋았지만","스토리는 너무 지루했다"],"brief_reason":"긍정 표현보다 스토리에 대한 부정 평가가 최종 판단을 더 강하게 결정한다.","created_at":"2026-07-01"}
```

주의:

```text
1. model_id는 실제 호출한 모델명을 저장한다.
2. prompt_version을 반드시 저장한다.
3. temperature는 가능하면 0에 가깝게 둔다.
4. 같은 sample에 대한 응답은 캐시하여 비용과 변동성을 줄인다.
5. API를 쓰지 못하면 Claude UI에서 수동으로 얻은 결과를 같은 JSONL 형식에 맞춰 저장한다.
```

## 8. Opus와 XAI 비교 지표

### 8.1 Phrase Overlap Score

가장 단순한 지표는 Opus가 언급한 표현과 XAI Top-k 표현이 얼마나 겹치는지 보는 것이다.

```text
Overlap Score = 겹친 Opus evidence 수 / Opus evidence 전체 수
```

예:

```text
Opus evidence:
["스토리", "너무 지루했다"]

CNN XAI Top-3:
["너무 지루했다", "스토리", "좋았지만"]

Overlap Score:
2 / 2 = 1.0
```

### 8.2 Token Overlap Score

형태소/서브워드 차이를 줄이기 위해 표현을 token으로 분해한 뒤 비교한다.

```text
Opus tokens = normalize("스토리는 너무 지루했다")
XAI tokens = normalize("너무 지루하다")

Token Overlap = |intersection| / |Opus token set|
```

한국어에서는 조사와 어미가 달라질 수 있으므로, 비교용 normalize는 다음을 수행한다.

```text
1. Okt 형태소 분석
2. stopword 제거
3. stem=True 적용
4. 공백/기호 제거
```

### 8.3 Jaccard Similarity

Opus evidence token set과 XAI top-k token set의 대칭적 유사도를 본다.

```text
Jaccard = |A intersection B| / |A union B|
```

장점:

```text
XAI가 너무 많은 표현을 뽑아도 점수가 과하게 높아지는 것을 줄인다.
```

### 8.4 Polarity Agreement

Opus sentiment와 small model prediction이 같은지 본다.

```text
polarity_agreement = opus_sentiment == model_pred_label
```

이 지표는 XAI 설명 유사도와 따로 봐야 한다.

예:

```text
모델 예측은 Opus와 같지만, 근거 표현은 다를 수 있다.
모델 예측은 Opus와 다르지만, 일부 근거 표현은 겹칠 수 있다.
```

### 8.5 Explanation Alignment Score

최종 비교표에서는 다음 조합을 사용한다.

```text
Explanation Alignment Score =
0.5 * Token Overlap
+ 0.3 * Phrase Overlap
+ 0.2 * Jaccard
```

이 점수는 엄밀한 정답 지표가 아니라 발표용 비교 지표이다. 보고서에서는 "정성 비교를 보조하기 위한 단순 정량 지표"라고 표현한다.

## 9. Opus 비교 코드 구조

### 9.1 `OpusBaseline/prompt_templates.md`

역할:

```text
1. prompt version 관리
2. 한국어 감정 분석 prompt 보관
3. JSON 출력 schema 기록
4. 발표용 설명 문장 보관
```

### 9.2 `OpusBaseline/opus_prompt_builder.py`

역할:

```text
selected_reviews_20_30.csv를 읽고 Opus 요청 prompt를 만든다.
```

입력:

```text
sample_id, text
```

출력:

```text
sample_id, prompt_version, prompt
```

### 9.3 `OpusBaseline/collect_opus_explanations.py`

역할:

```text
Opus API 또는 수동 입력 파일을 통해 explanation jsonl을 만든다.
```

두 가지 mode를 둔다.

```powershell
python XAI\OpusBaseline\collect_opus_explanations.py --mode api
python XAI\OpusBaseline\collect_opus_explanations.py --mode manual --input XAI\comparison_outputs\opus_manual.csv
```

API mode 주의:

```text
ANTHROPIC_API_KEY 환경변수가 없으면 실행하지 않는다.
응답은 항상 jsonl cache에 저장한다.
이미 sample_id 응답이 있으면 재호출하지 않는다.
```

### 9.4 `OpusBaseline\normalize_opus_outputs.py`

역할:

```text
Opus evidence phrase를 비교 가능한 token set으로 변환한다.
```

출력:

```text
XAI/comparison_outputs/opus_explanations_normalized.csv
```

컬럼:

```text
sample_id
opus_sentiment
opus_evidence_phrase
opus_evidence_tokens
brief_reason
```

### 9.5 `OpusBaseline/compare_opus_with_xai.py`

역할:

```text
xai_topk_unified.csv와 opus_explanations_normalized.csv를 결합하여 overlap score를 계산한다.
```

출력:

```text
XAI/comparison_outputs/opus_xai_overlap_scores.csv
XAI/comparison_outputs/qualitative_case_report.md
```

## 10. Opus 비교 notebook 계획

### 10.1 `01_opus_prompt_preview.ipynb`

목적:

```text
Opus에 보낼 20~30개 prompt를 사람이 먼저 검토한다.
```

포함 셀:

```text
1. selected_reviews_20_30.csv 로드
2. prompt 생성
3. sample_type별 prompt 미리보기
4. prompt version 저장
```

### 10.2 `02_opus_xai_comparison.ipynb`

목적:

```text
Opus explanation과 FNN/CNN/Transformer XAI top-k를 한 화면에서 비교한다.
```

포함 셀:

```text
1. model_predictions_unified.csv 로드
2. xai_topk_unified.csv 로드
3. opus_explanations.jsonl 로드
4. sample_id 선택 widget
5. 모델별 예측 확률 표
6. 모델별 XAI top-k evidence 표
7. Opus evidence와 brief_reason 표시
8. overlap score bar chart
```

## 11. 최종 비교 표 예시

문장:

```text
배우는 좋았지만 스토리는 너무 지루했다.
```

예상 비교:

| 모델 | 예측 | 확률 | 중요 근거 |
|---|---|---:|---|
| FNN | 긍정 | 0.55 | `좋았지만`, `배우`, `지루했다` |
| CNN | 부정 | 0.72 | `너무 지루했다`, `스토리 너무 지루했다` |
| Transformer | 부정 | 0.84 | `스토리`, `너무`, `지루했다` |
| Opus | 부정 | - | `좋았지만`, `스토리는 너무 지루했다` |

해석 예시:

```text
Opus는 긍정 표현인 "좋았지만"보다 "스토리는 너무 지루했다"라는 최종 평가 구간을 더 강한 부정 근거로 보았다.
CNN과 Transformer는 모두 "너무 지루했다" 계열 표현을 중요 근거로 제시하여 Opus 설명과 높은 겹침을 보였다.
반면 FNN은 단어 단위 반응이 강해 "좋았지만"에 상대적으로 높은 중요도를 부여하여 문맥 반전 처리에서 한계를 보였다.
```

## 12. 보고서 구조 개선안

기존 구조:

```text
1. FNN
2. CNN
3. Transformer
```

개선 구조:

```text
1. 프로젝트 목적
   - NSMC 감정 분석 모델의 판단 근거 비교
   - 작은 모델 XAI와 고성능 LLM 자연어 설명의 비교

2. 모델 설명
   - FNN: 단어 중심
   - CNN: n-gram/filter 중심
   - Transformer: 문맥 기반
   - Opus: 정성적 기준 설명

3. 모델 성능 비교
   - Accuracy, F1, confusion matrix

4. XAI 방법
   - Occlusion
   - n-gram Occlusion
   - Gradient x Input / Integrated Gradients
   - Attention은 Transformer 보조 분석으로만 사용

5. 모델별 XAI 결과
   - FNN evidence
   - CNN evidence
   - Transformer evidence

6. Opus 기준 설명과 비교
   - Opus prompt
   - Opus evidence
   - overlap score
   - 대표 사례 정성 분석

7. 실패 사례 분석
   - 작은 모델 오분류
   - Opus 설명과 어긋난 evidence
   - 반전 표현/부정어/복합 감정 문장

8. 한계
   - Opus 설명은 정답이 아님
   - LLM explanation은 사후 자연어 설명
   - XAI도 근사 설명
   - tokenization 단위 차이

9. 결론
   - 모델 구조가 evidence 단위에 미치는 영향
   - CNN은 구절 중심, Transformer는 문맥 중심 evidence가 강함
   - Opus 기준과의 비교는 정성적 해석을 보강함
```

## 13. 구현 단계

### Phase 1. 현재 CNN 통합 스크립트 안정화

상태:

```text
완료됨.
현재 nsmc_cnn_xai.py는 전체 CNN XAI 산출물을 생성한다.
```

유지할 것:

```text
1. 기존 실행 명령
2. xai_outputs 산출물 이름
3. cnn_preprocess_cache.pkl
```

### Phase 2. shared 모듈 생성

상태:

```text
완료됨.
XAI/CNN/shared 아래에 경로, 데이터 로드, 토큰화, schema, 시각화 helper를 분리하였다.
```

작업:

```text
1. paths.py
2. nsmc_data.py
3. tokenization.py
4. schemas.py
5. visualization.py
```

검증:

```text
python -m py_compile XAI\CNN\shared\*.py
python XAI\CNN\nsmc_cnn_xai.py --only-custom --sample-size 0 --pool-size 0
```

### Phase 3. CNN 방법별 모듈 분리

상태:

```text
완료됨.
기존 XAI/CNN/nsmc_cnn_xai.py는 compatibility wrapper로 유지하고,
실제 구현은 XAI/CNN/xai_methods 아래 방법별 모듈로 분리하였다.
Integrated Gradients도 embedding-space PyTorch 구현으로 추가하였다.
```

작업:

```text
1. model.py
2. loader.py
3. unigram_occlusion.py
4. ngram_occlusion.py
5. filter_activation.py
6. maxpool_positions.py
7. gradient_x_input.py
8. integrated_gradients.py
9. run_all.py
```

검증:

```text
python XAI\CNN\xai_methods\run_all.py --only-custom --sample-size 0 --pool-size 0 --output-dir XAI\CNN\xai_outputs_smoke
python XAI\CNN\xai_methods\run_all.py --sample-size 300 --pool-size 1000 --output-dir XAI\CNN\xai_outputs
```

### Phase 4. CNN notebook 생성

상태:

```text
완료됨.
XAI/CNN/notebooks 아래에 CSV 산출물을 읽어 방법별 결과를 빠르게 확인하는 notebook 7개를 생성하였다.
노트북은 대량 재계산 없이 xai_outputs 파일을 읽는 해석/점검 용도이다.
```

작업:

```text
1. notebooks/00_cnn_xai_overview.ipynb
2. notebooks/01_cnn_unigram_occlusion.ipynb
3. notebooks/02_cnn_ngram_occlusion.ipynb
4. notebooks/03_cnn_filter_activation.ipynb
5. notebooks/04_cnn_maxpool_positions.ipynb
6. notebooks/05_cnn_gradient_x_input.ipynb
7. notebooks/06_cnn_integrated_gradients.ipynb
```

검증:

```text
각 notebook이 CSV를 읽고 최소 1개 표/그림을 출력한다.
노트북 안에서 대량 재계산을 하지 않는다.
```

### Phase 5. 통합 sample set 생성

작업:

```text
1. selected_reviews_20_30.csv 생성
2. CNN selected sample 중 대표 사례 재사용
3. FNN/Transformer에서도 같은 sample_id를 사용
```

검증:

```text
FNN/CNN/Transformer/Opus 모두 같은 sample_id로 join 가능해야 한다.
```

### Phase 6. FNN/Transformer XAI 결과 통합

작업:

```text
1. FNN XAI 결과를 xai_topk_unified.csv 형식으로 변환
2. CNN XAI 결과를 xai_topk_unified.csv 형식으로 변환
3. Transformer XAI 결과를 xai_topk_unified.csv 형식으로 변환
```

검증:

```text
model_name in [fnn, cnn, transformer]
method가 비어 있지 않음
evidence_text가 비어 있지 않음
sample_id 기준으로 모델별 top-k 비교 가능
```

### Phase 7. Opus baseline 추가

작업:

```text
1. prompt_templates.md 작성
2. opus_prompt_builder.py 작성
3. collect_opus_explanations.py 작성
4. opus_explanations.jsonl 생성
5. normalize_opus_outputs.py 작성
```

검증:

```text
20~30개 sample에 대해 sentiment, evidence_phrases, brief_reason이 모두 존재한다.
JSON parsing 실패 샘플은 별도 error file에 저장한다.
```

### Phase 8. Opus-XAI overlap 계산

작업:

```text
1. compare_opus_with_xai.py 작성
2. phrase overlap, token overlap, jaccard 계산
3. 모델별 평균 score 산출
4. qualitative_case_report.md 생성
```

검증:

```text
FNN/CNN/Transformer별 average overlap score 출력
sample별 가장 Opus와 가까운 모델 출력
오분류/불일치 사례 5개 이상 정리
```

## 14. 최종 산출물

코드 산출물:

```text
XAI/CNN/shared/*.py
XAI/CNN/xai_methods/*.py
XAI/CNN/notebooks/*.ipynb
XAI/OpusBaseline/*.py
XAI/OpusBaseline/notebooks/*.ipynb
```

분석 산출물:

```text
XAI/CNN/xai_outputs/*.csv
XAI/CNN/xai_outputs/figures/*.png
XAI/comparison_outputs/selected_reviews_20_30.csv
XAI/comparison_outputs/model_predictions_unified.csv
XAI/comparison_outputs/xai_topk_unified.csv
XAI/comparison_outputs/opus_explanations.jsonl
XAI/comparison_outputs/opus_explanations_normalized.csv
XAI/comparison_outputs/opus_xai_overlap_scores.csv
XAI/comparison_outputs/qualitative_case_report.md
```

보고서 산출물:

```text
1. 모델별 성능 비교표
2. 모델별 XAI evidence 비교표
3. Opus 기준 overlap score 표
4. 대표 사례 분석 4~6개
5. 한계와 해석 주의점
```

## 15. 발표용 핵심 문장

```text
추가적으로 고성능 LLM인 Opus를 정성적 기준 모델로 활용하여, 각 리뷰에 대한 자연어 판단 근거를 생성한다.
이후 FNN, CNN, Transformer의 XAI 결과와 Opus의 판단 근거를 비교함으로써,
작은 딥러닝 모델의 의사결정 근거가 고성능 언어모델의 자연어 해석과 얼마나 유사한지 분석한다.
```

```text
Opus 기준 분석은 정답 라벨을 대체하지 않는다.
본 프로젝트에서는 Opus의 설명을 사람이 이해하기 쉬운 참고 기준으로 사용하고,
Occlusion, Gradient, Attention 등 모델 내부/입력 기반 XAI 결과와의 유사성을 정성적으로 비교한다.
```

```text
CNN은 n-gram occlusion과 filter activation에서 구절 중심 evidence를 보였고,
Transformer는 반전 표현과 문맥 단서에서 Opus 설명과 더 높은 alignment를 보이는지 확인한다.
```

## 16. 참고 자료 기록

Opus 관련 모델명과 권장 용도는 시간이 지나며 바뀔 수 있으므로, 보고서 작성 직전에 Anthropic 공식 문서를 다시 확인한다.

계획 작성 시 참고한 공식 문서:

```text
Anthropic Docs - Models overview
https://docs.anthropic.com/en/docs/about-claude/models/overview

Anthropic Docs - Claude 4
https://docs.anthropic.com/en/docs/about-claude/models/claude-4
```
