# Opus 기준 XAI 비교 구현계획서

## 1. 목표

작은 감정분류 모델의 XAI 결과가 고성능 LLM인 Opus의 사후 자연어 판단 근거와 얼마나 유사한지 비교한다. 비교 대상은 단순히 모델 단위가 아니라 `모델 x XAI 기법` 단위로 둔다.

핵심 목표는 다음과 같다.

1. FNN, CNN, Transformer 중 어떤 모델의 중요 표현이 Opus 근거와 가장 많이 겹치는지 확인한다.
2. 각 모델 내부에서 어떤 XAI 기법이 Opus 근거와 가장 잘 맞는지 비교한다.
3. CNN의 n-gram evidence가 Opus의 구절 중심 근거와 잘 맞는지 확인한다.
4. Transformer의 문맥 기반 XAI가 반전 표현에서 Opus 근거와 더 잘 맞는지 확인한다.
5. 오분류 사례에서 작은 모델이 Opus와 다르게 본 표현을 정리한다.

주의: Opus 설명은 정답 라벨이나 절대 기준이 아니다. 이 실험에서는 사람이 읽기 쉬운 고성능 LLM의 정성적 기준 설명으로만 사용한다.

## 2. 최종 산출물

```text
XAI/
  OPUS_XAI_COMPARISON_IMPLEMENTATION_PLAN.md
  OpusBaseline/
    prompt_templates.md
    opus_prompt_builder.py
    collect_opus_explanations.py
    normalize_opus_outputs.py
    unify_xai_outputs.py
    compare_opus_with_xai.py
    comparison_outputs/
      selected_reviews.csv
      opus_prompts.jsonl
      opus_explanations.jsonl
      xai_unified.jsonl
      opus_vectors.jsonl
      opus_xai_overlap_scores.csv
      opus_xai_method_summary.csv
      qualitative_case_report.md
```

## 3. 입력 데이터

### 3.1 분석 샘플

우선 현재 `inputs.txt`의 10문장을 사용한다. 이후 보고서용으로 20~30문장으로 확장한다.

권장 샘플 구성:

| 유형 | 개수 | 목적 |
| --- | ---: | --- |
| 명확한 긍정 | 5 | 쉬운 긍정 근거 비교 |
| 명확한 부정 | 5 | 쉬운 부정 근거 비교 |
| 긍정+부정 혼합 | 5 | 상반된 evidence 처리 비교 |
| 반전 표현 | 5 | `하지만`, `인데`, `갈수록`, `처음엔` 같은 문맥 처리 비교 |
| 모델 오분류 사례 | 5~10 | Opus와 작은 모델이 다르게 본 표현 분석 |

저장 형식:

```csv
sample_id,text,true_label,sample_type,selection_reason
case_001,배우들의 연기는 명품인데 각본이 너무 쓰레기네요,negative,mixed,positive acting but negative script
```

### 3.2 XAI 결과

모든 모델의 XAI 결과는 최종적으로 어절 단위 JSONL로 통합한다.

공통 형식:

```json
{
  "sample_id": "case_001",
  "model": "cnn",
  "method": "ngram_occlusion",
  "text": "배우들의 연기는 명품인데 각본이 너무 쓰레기네요",
  "prediction": "negative",
  "probability": 0.9859,
  "words": ["배우들의", "연기는", "명품인데", "각본이", "너무", "쓰레기네요"],
  "scores": [0.0123, 0.0045, 0.1502, -0.2104, -0.1201, -0.5025],
  "score_target": "positive",
  "score_normalization": "word_l1",
  "source_file": "XAI/CNN/outputs/output_cnn_ngram_occlusion.json"
}
```

해석 규칙:

- `scores`는 모두 어절 단위이다.
- L1 정규화를 적용한다.
- 양수는 positive class 기여, 음수는 negative class 기여로 해석한다.
- raw score 의미는 기법마다 다르므로, 비교 지표에서는 normalized score를 쓰되 보고서에는 기법별 의미 차이를 설명한다.

## 4. Opus 결과 수집 방식

### 4.1 Prompt 원칙

Opus에게 XAI 결과를 보여주지 않는다. Opus는 원문과 어절 리스트만 보고 독립적으로 판단하게 한다.

Opus에게 반드시 요구할 것:

1. 감정 라벨을 `positive` 또는 `negative`로 출력한다.
2. 판단 근거를 원문 어절 index 기준으로 반환한다.
3. 근거별 polarity와 strength를 반환한다.
4. JSON만 출력하게 한다.

### 4.2 Prompt 입력 예시

```json
{
  "sample_id": "case_001",
  "text": "배우들의 연기는 명품인데 각본이 너무 쓰레기네요",
  "words": ["배우들의", "연기는", "명품인데", "각본이", "너무", "쓰레기네요"]
}
```

### 4.3 Opus 출력 schema

```json
{
  "sample_id": "case_001",
  "model_id": "실제 호출한 Opus model id",
  "prompt_version": "opus_sentiment_evidence_v1",
  "text": "배우들의 연기는 명품인데 각본이 너무 쓰레기네요",
  "words": ["배우들의", "연기는", "명품인데", "각본이", "너무", "쓰레기네요"],
  "sentiment": "negative",
  "evidence": [
    {
      "phrase": "각본이 너무 쓰레기네요",
      "word_indices": [3, 4, 5],
      "polarity": "negative",
      "strength": 0.95,
      "reason": "각본에 대한 강한 부정 평가가 최종 감정을 결정한다."
    },
    {
      "phrase": "연기는 명품인데",
      "word_indices": [1, 2],
      "polarity": "positive",
      "strength": 0.35,
      "reason": "배우 연기에 대한 긍정 표현이지만 최종 감정을 뒤집지는 못한다."
    }
  ],
  "brief_reason": "배우 연기는 긍정적이지만 각본에 대한 강한 부정 표현이 최종 감정을 지배한다."
}
```

## 5. 비교 지표

### 5.1 Opus evidence vector

Opus evidence를 XAI와 같은 길이의 어절 vector로 변환한다.

규칙:

```text
positive evidence: +strength
negative evidence: -strength
같은 어절에 여러 evidence가 겹치면 합산
마지막에 L1 정규화
```

예시:

```text
words = ["배우들의", "연기는", "명품인데", "각본이", "너무", "쓰레기네요"]
opus_vector = [0.0, 0.1346, 0.1346, -0.2436, -0.2436, -0.2436]
```

### 5.2 Top-k Recall

Opus가 evidence로 찍은 어절 중 XAI top-k 어절에 포함된 비율이다.

```text
topk_recall = |Opus evidence indices ∩ XAI top-k indices| / |Opus evidence indices|
```

권장 k:

```text
k = min(3, 문장 어절 수)
```

### 5.3 Jaccard Similarity

Opus evidence set과 XAI top-k set의 대칭적 겹침을 본다.

```text
jaccard = |A ∩ B| / |A ∪ B|
```

### 5.4 Evidence Mass

Opus evidence 위치에 XAI 중요도가 얼마나 몰렸는지 본다.

```text
evidence_mass = sum(abs(xai_score[i]) for i in opus_indices)
```

XAI score가 L1 정규화되어 있으면 0~1 사이 값이 된다.

### 5.5 Signed Cosine Similarity

긍정/부정 방향까지 포함해 Opus vector와 XAI vector의 유사도를 본다.

```text
signed_cosine = cosine(opus_vector, xai_vector)
```

해석:

- 1에 가까움: 중요 어절과 polarity 방향이 모두 유사함
- 0에 가까움: 근거 위치나 방향이 거의 무관함
- 음수: Opus와 반대 방향으로 해석함

### 5.6 Evidence Polarity Agreement

겹친 어절에서 Opus polarity와 XAI score 부호가 같은 비율이다.

```text
polarity_agreement =
  same_sign_count / overlapping_evidence_count
```

### 5.7 종합 점수

발표용 요약 지표로만 사용한다.

```text
opus_match_score =
  0.35 * topk_recall
+ 0.25 * jaccard
+ 0.25 * max(0, signed_cosine)
+ 0.15 * evidence_mass
```

주의: 이 점수는 편의상 만든 비교 지표이다. 보고서에서는 개별 지표와 함께 해석한다.

## 6. 모델 x 기법 비교 설계

최종 비교 단위는 다음이다.

```text
sample_id, model, method
```

최종 CSV schema:

```csv
sample_id,model,method,prediction,probability,opus_sentiment,topk_recall,jaccard,signed_cosine,evidence_mass,polarity_agreement,opus_match_score
```

집계 테이블:

```csv
model,method,n_samples,mean_topk_recall,mean_jaccard,mean_signed_cosine,mean_evidence_mass,mean_opus_match_score,rank
```

분석 질문별 집계:

| 질문 | 필터/그룹 |
| --- | --- |
| 어떤 모델이 Opus와 가장 가까운가? | `groupby(model)` |
| 각 모델 내부에서 어떤 XAI 기법이 가장 가까운가? | `groupby(model, method)` |
| CNN n-gram evidence가 잘 맞는가? | `model == cnn`, `method == ngram_occlusion` |
| 반전 표현에서 Transformer가 강한가? | `sample_type == contrast`, `model == transformer` |
| 오분류에서 무엇이 달랐는가? | `prediction != opus_sentiment` |

## 7. 구현 파일별 계획

### 7.1 `XAI/OpusBaseline/prompt_templates.md`

역할:

- Opus prompt version 관리
- JSON 출력 schema 명시
- 보고서에 들어갈 prompt 설명 보관

포함 내용:

- `opus_sentiment_evidence_v1`
- system/user prompt
- 실패 응답 예시와 금지 규칙

### 7.2 `XAI/OpusBaseline/opus_prompt_builder.py`

역할:

- `selected_reviews.csv`를 읽는다.
- 각 문장을 `words = text.split()`으로 나눈다.
- Opus 요청용 prompt JSONL을 만든다.

입력:

```text
XAI/OpusBaseline/comparison_outputs/selected_reviews.csv
```

출력:

```text
XAI/OpusBaseline/comparison_outputs/opus_prompts.jsonl
```

### 7.3 `XAI/OpusBaseline/collect_opus_explanations.py`

역할:

- API mode 또는 manual mode로 Opus 결과를 수집한다.
- 이미 존재하는 `sample_id`는 다시 호출하지 않는다.
- 실제 `model_id`, `prompt_version`, `created_at`을 저장한다.

실행 예:

```powershell
python XAI/OpusBaseline/collect_opus_explanations.py --mode manual --input XAI/OpusBaseline/comparison_outputs/opus_manual.jsonl
python XAI/OpusBaseline/collect_opus_explanations.py --mode api --prompts XAI/OpusBaseline/comparison_outputs/opus_prompts.jsonl
```

처음 구현은 manual mode부터 만든다. API mode는 나중에 붙여도 된다.

### 7.4 `XAI/OpusBaseline/unify_xai_outputs.py`

역할:

- FNN/CNN/Transformer의 JSON 결과를 공통 JSONL로 변환한다.
- 각 row에 `sample_id`, `model`, `method`, `scores`를 붙인다.
- method별 score key 차이를 흡수한다.

CNN 입력:

```text
XAI/CNN/outputs/output_cnn_unigram_occlusion.json
XAI/CNN/outputs/output_cnn_ngram_occlusion.json
XAI/CNN/outputs/output_cnn_filter_activation.json
XAI/CNN/outputs/output_cnn_integrated_gradients.json
```

출력:

```text
XAI/OpusBaseline/comparison_outputs/xai_unified.jsonl
```

구현 시 우선 CNN만 지원하고, 이후 FNN/Transformer JSON이 준비되면 loader를 추가한다.

### 7.5 `XAI/OpusBaseline/normalize_opus_outputs.py`

역할:

- Opus JSONL을 검증한다.
- `word_indices`가 범위를 벗어나지 않는지 확인한다.
- Opus evidence vector를 만든다.

출력:

```text
XAI/OpusBaseline/comparison_outputs/opus_vectors.jsonl
```

검증 규칙:

- `sentiment`는 `positive|negative`
- 모든 `word_indices`는 `0 <= idx < len(words)`
- evidence가 최소 1개 이상 존재
- evidence phrase가 비어 있지 않음

### 7.6 `XAI/OpusBaseline/compare_opus_with_xai.py`

역할:

- `xai_unified.jsonl`과 `opus_vectors.jsonl`을 `sample_id` 기준으로 join한다.
- `model x method x sample` 단위 지표를 계산한다.
- 전체/샘플 유형별/오분류별 summary를 만든다.

출력:

```text
XAI/OpusBaseline/comparison_outputs/opus_xai_overlap_scores.csv
XAI/OpusBaseline/comparison_outputs/opus_xai_method_summary.csv
XAI/OpusBaseline/comparison_outputs/qualitative_case_report.md
```

## 8. 정성 분석 report 구성

`qualitative_case_report.md`는 다음 형태로 만든다.

```markdown
## case_001

text: 배우들의 연기는 명품인데 각본이 너무 쓰레기네요

Opus:
- sentiment: negative
- evidence: 각본이 너무 쓰레기네요
- reason: 배우 연기는 긍정적이지만 각본 부정 표현이 최종 감정을 지배한다.

Top XAI evidence:
| model | method | prediction | top words | opus_match_score |
| --- | --- | --- | --- | ---: |
| CNN | ngram_occlusion | negative | 쓰레기네요, 너무, 각본이 | 0.82 |
| CNN | integrated_gradients | negative | 쓰레기네요, 너무, 명품인데 | 0.68 |

해석:
- CNN n-gram occlusion은 Opus가 본 핵심 부정 구간과 잘 겹친다.
- Integrated Gradients는 부정 어절도 잡지만 긍정 표현 일부에도 attribution이 분산된다.
```

## 9. 구현 순서

### Phase 1. 샘플과 CNN 기준 pipeline 고정

1. `selected_reviews.csv` 생성
2. CNN JSON 4종을 현재 `export_word_json.py`로 생성
3. `unify_xai_outputs.py`에서 CNN 4종을 `xai_unified.jsonl`로 변환
4. `sample_id` 매핑 규칙 확정

완료 기준:

- CNN 4개 method가 모두 `xai_unified.jsonl`에 들어간다.
- 각 row의 `words`와 `scores` 길이가 같다.

### Phase 2. Opus manual 수집 형식 구현

1. `prompt_templates.md` 작성
2. `opus_prompt_builder.py` 작성
3. 사람이 복사/붙여넣기할 수 있는 prompt preview 생성
4. manual JSONL 입력을 받아 `opus_explanations.jsonl` 생성

완료 기준:

- 10개 샘플에 대해 Opus JSONL이 생성된다.
- schema validation이 통과한다.

### Phase 3. Opus vector와 비교 지표 구현

1. `normalize_opus_outputs.py` 작성
2. `compare_opus_with_xai.py` 작성
3. 지표 CSV와 method summary CSV 생성

완료 기준:

- `sample_id, model, method` 단위 score가 생성된다.
- `groupby(model, method)` summary가 생성된다.

### Phase 4. FNN/Transformer 확장

1. FNN 결과를 어절 단위 JSON으로 export
2. Transformer 결과를 어절 단위 JSON으로 export
3. `unify_xai_outputs.py`에 loader 추가
4. 전체 모델 비교 재실행

완료 기준:

- FNN/CNN/Transformer가 같은 `xai_unified.jsonl` schema를 사용한다.
- 모델별, 기법별 summary가 모두 생성된다.

### Phase 5. 보고서용 정성 사례 작성

1. match score 상위 사례 3개
2. match score 하위 사례 3개
3. 반전 표현 사례 3개
4. 오분류 사례 3개

완료 기준:

- `qualitative_case_report.md`가 생성된다.
- 각 사례에 Opus evidence와 모델별 top evidence가 함께 표시된다.

## 10. 우선순위

1. CNN 4개 method만으로 end-to-end 비교를 먼저 완성한다.
2. Opus 수집은 manual mode부터 구현한다.
3. 지표 계산과 report 생성이 안정화된 뒤 FNN/Transformer를 붙인다.
4. API 호출 자동화는 마지막에 붙인다.

## 11. 리스크와 대응

| 리스크 | 대응 |
| --- | --- |
| Opus가 원문에 없는 표현을 evidence로 만든다 | `word_indices` 필수 출력으로 제한 |
| Opus 응답 형식이 깨진다 | JSON schema validation과 manual correction 파일 제공 |
| 모델별 tokenizer가 달라 직접 비교가 어렵다 | 어절 단위로 합산 후 비교 |
| XAI method별 score 의미가 다르다 | raw score 의미는 보존하고, 비교는 L1 normalized score로 수행 |
| Opus와 모델 prediction이 다르다 | prediction agreement와 explanation overlap을 분리해 해석 |
| 종합 점수가 과도하게 권위 있어 보인다 | 보고서에서 보조 지표라고 명시 |

## 12. 보고서 표현 가이드

사용하기 좋은 표현:

```text
Opus 설명은 정답이 아니라 고성능 LLM의 사후 자연어 판단 기준으로 사용하였다.
본 실험은 작은 모델의 XAI 결과와 Opus가 언급한 핵심 표현의 표면적 겹침과 polarity 방향성을 비교한다.
```

피해야 할 표현:

```text
Opus가 정답이다.
Transformer는 Opus와 같은 방식으로 생각했다.
CNN 설명은 Opus와 다르므로 틀렸다.
```

## 13. 1차 구현 완료 조건

1차 구현은 CNN만 대상으로 한다.

완료 조건:

- `XAI/OpusBaseline/comparison_outputs/xai_unified.jsonl` 생성
- `XAI/OpusBaseline/comparison_outputs/opus_explanations.jsonl` 생성
- `XAI/OpusBaseline/comparison_outputs/opus_vectors.jsonl` 생성
- `XAI/OpusBaseline/comparison_outputs/opus_xai_overlap_scores.csv` 생성
- `XAI/OpusBaseline/comparison_outputs/opus_xai_method_summary.csv` 생성
- CNN 4개 method 중 Opus와 가장 잘 맞는 method를 평균 점수로 확인 가능

이후 FNN/Transformer 결과를 같은 schema로 추가하면 전체 실험 7로 확장한다.
