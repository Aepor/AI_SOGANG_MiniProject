# OpusBaseline

Opus의 사후 자연어 판단 근거와 XAI 결과를 비교하는 1차 파이프라인입니다.

## 1. Prompt 생성

```powershell
python XAI/OpusBaseline/opus_prompt_builder.py --input-file inputs.txt
```

생성 파일:

- `XAI/OpusBaseline/comparison_outputs/selected_reviews.csv`
- `XAI/OpusBaseline/comparison_outputs/opus_prompts.jsonl`

`opus_prompts.jsonl`의 `prompt`를 Opus에 보내고, 응답 JSON을 한 줄에 하나씩 `XAI/OpusBaseline/comparison_outputs/opus_manual.jsonl`에 저장합니다.

## 2. XAI 결과 통합

```powershell
python XAI/OpusBaseline/unify_xai_outputs.py
```

생성 파일:

- `XAI/OpusBaseline/comparison_outputs/xai_unified.jsonl`

현재는 CNN 4개 method를 지원합니다.

## 3. Opus 응답 수집 및 정규화

```powershell
python XAI/OpusBaseline/collect_opus_explanations.py --mode manual
python XAI/OpusBaseline/normalize_opus_outputs.py
```

생성 파일:

- `XAI/OpusBaseline/comparison_outputs/opus_explanations.jsonl`
- `XAI/OpusBaseline/comparison_outputs/opus_vectors.jsonl`

## 4. 비교 지표 계산

```powershell
python XAI/OpusBaseline/compare_opus_with_xai.py
```

생성 파일:

- `XAI/OpusBaseline/comparison_outputs/opus_xai_overlap_scores.csv`
- `XAI/OpusBaseline/comparison_outputs/opus_xai_method_summary.csv`
- `XAI/OpusBaseline/comparison_outputs/qualitative_case_report.md`

## Manual Opus JSONL 형식

```json
{"sample_id":"case_001","sentiment":"negative","evidence":[{"phrase":"...","word_indices":[1,2],"polarity":"negative","strength":0.9,"reason":"..."}],"brief_reason":"..."}
```

`text`와 `words`는 생략해도 `opus_prompts.jsonl`에서 자동으로 보강됩니다.
