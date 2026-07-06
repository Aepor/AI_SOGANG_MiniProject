# CNN XAI 정리

이 폴더는 NSMC CNN 모델과 발표/비교용 XAI JSON export만 남긴 구조입니다.

## 핵심 실행

```powershell
python XAI/CNN/export_word_json.py --input-file inputs.txt
```

기본 JSON 출력 위치는 `XAI/outputs_json/`이고, 그래프 출력 위치는 `XAI/outputs_graph/cnn_*/`입니다.

## 남긴 XAI 방법

- `unigram_occlusion`: 형태소 token 하나를 `<pad>`로 바꿨을 때 positive class 점수가 얼마나 변하는지 계산합니다.
- `ngram_occlusion`: 연속 token window를 `<pad>`로 바꿨을 때 positive class 점수가 얼마나 변하는지 계산합니다.
- `filter_activation`: CNN filter가 강하게 반응한 n-gram의 classifier contribution을 계산합니다.
- `integrated_gradients`: all-`<pad>` baseline에서 입력 embedding까지의 경로 attribution을 계산합니다.

모든 JSON score는 model token 점수를 어절 단위로 합산한 뒤 L1 정규화합니다. 양수는 positive class에 기여, 음수는 negative class 방향 기여로 해석하면 됩니다.

## 폴더 구조

```text
XAI/CNN/
  export_word_json.py        # inputs.txt -> output_cnn_*.json
  nsmc_cnn_xai.py            # 예전 실행 경로 호환 wrapper
  nsmc_cnn.ipynb             # CNN 학습 notebook
  best_cnn_model.pt          # 학습된 CNN checkpoint
  cnn_preprocess_cache.pkl   # checkpoint와 맞춘 vocab cache
  metrics.json               # CNN 평가 지표
  xai_methods/               # 네 가지 XAI 방법 + model/loader
  shared/                    # 경로, 데이터, tokenization, schema helper
  docs/                      # 설명 문서와 이전 분석 계획
```
