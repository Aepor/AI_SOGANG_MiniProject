# LLMComparison

OpenAI API로 수집한 LLM의 사후 자연어 판단 근거와 XAI 결과를 비교하는 1차 파이프라인입니다.

API 키는 repo root의 `.env`에서 읽습니다. `.env`는 git에 올라가지 않습니다.

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5-pro
OPENAI_REASONING_EFFORT=high
```

`gpt-5.5-pro`는 성능 우선 설정입니다. 요청이 느릴 수 있으므로 먼저 작은 배치로 응답 형식을 확인합니다.

자세한 구조 설명은 `XAI/LLMComparison/LLM_COMPARISON_PIPELINE_GUIDE.md`에 정리되어 있습니다.

## 전체 파이프라인 실행

앞으로 실험은 아래 명령 하나로 실행합니다.

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py --max-samples 5
```

전체 샘플을 실행할 때는 `--max-samples`를 생략합니다.

```powershell
python XAI/LLMComparison/run_llm_xai_experiment.py
```

파이프라인 단계:

1. CNN/FNN/Transformer XAI 결과를 `xai_unified.jsonl`로 통합
2. LLM 입력 prompt 생성
3. OpenAI API로 LLM evidence 수집
4. LLM evidence를 어절 vector로 정규화
5. LLM evidence와 XAI 결과를 비교하고 평가 보고서 생성

## 1. Prompt 생성

```powershell
python XAI/LLMComparison/llm_prompt_builder.py --input-file inputs.txt
```

생성 파일:

- `XAI/LLMComparison/comparison_outputs/selected_reviews.csv`
- `XAI/LLMComparison/comparison_outputs/llm_prompts.jsonl`

## 2. XAI 결과 통합

```powershell
python XAI/LLMComparison/unify_xai_outputs.py --models cnn,fnn,transformer
```

생성 파일:

- `XAI/LLMComparison/comparison_outputs/xai_unified.jsonl`

현재는 `XAI/outputs_json`의 CNN 5개 method, FNN 4개 method, Transformer 3개 method를 지원합니다. `output_transformer_attention.json`은 비교에서 제외합니다.

## 3. LLM 응답 수집 및 정규화

OpenAI API 사용:

```powershell
python XAI/LLMComparison/collect_llm_explanations.py --mode api --max-samples 5
python XAI/LLMComparison/normalize_llm_outputs.py
```

수동 JSONL 사용:

```powershell
python XAI/LLMComparison/collect_llm_explanations.py --mode manual
python XAI/LLMComparison/normalize_llm_outputs.py
```

생성 파일:

- `XAI/LLMComparison/comparison_outputs/llm_explanations.jsonl`
- `XAI/LLMComparison/comparison_outputs/llm_vectors.jsonl`

## 4. 비교 지표 계산

```powershell
python XAI/LLMComparison/compare_llm_with_xai.py
```

생성 파일:

- `XAI/LLMComparison/comparison_outputs/llm_xai_overlap_scores.csv`
- `XAI/LLMComparison/comparison_outputs/llm_xai_method_summary.csv`
- `XAI/LLMComparison/comparison_outputs/qualitative_case_report.md`
- `XAI/LLMComparison/comparison_outputs/llm_xai_evaluation_report.md`

## Manual LLM JSONL 형식

```json
{"sample_id":"case_001","sentiment":"negative","evidence":[{"phrase":"...","word_indices":[1,2],"polarity":"negative","strength":0.9,"reason":"..."}],"brief_reason":"..."}
```

`text`와 `words`는 생략해도 `llm_prompts.jsonl`에서 자동으로 보강됩니다.
