# KcELECTRA NSMC 감정분석

이 폴더는 `beomi/KcELECTRA-base` 모델을 사용해 NSMC 영화 리뷰를
긍정/부정으로 분류하기 위한 코드입니다.

지원 기능은 다음과 같습니다.

- `train`: NSMC 학습 데이터로 KcELECTRA fine-tuning
- `eval`: 저장된 모델을 test 데이터로 최종 평가
- `predict`: 입력 문장의 긍정/부정 예측

## 데이터 분리 방식

현재 구조는 연구/보고서용 평가를 위해 train, validation, test를 분리합니다.

```text
Data/NSMC/ratings_train.txt
  -> train split
  -> validation split

Data/NSMC/ratings_test.txt
  -> final test only
```

기본적으로 `ratings_train.txt`의 10%를 validation으로 나눕니다.
validation set은 학습 중 best model 선택에만 사용하고,
`ratings_test.txt`는 학습이 끝난 뒤 최종 평가용으로만 사용합니다.

## 가상환경 실행

PowerShell에서 가상환경을 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

가상환경을 활성화하지 않고 바로 실행하려면 아래처럼 `.venv`의 Python을 직접 사용해도 됩니다.

```powershell
.venv\Scripts\python.exe Transformer\kcelectra_nsmc.py --help
```

## 패키지 설치

```powershell
pip install -r requirements.txt
```

이미 `.venv`에 설치되어 있다면 다시 설치하지 않아도 됩니다.

## 학습

기본 학습 명령입니다.

```powershell
python Transformer/kcelectra_nsmc.py train `
  --train-file Data/NSMC/ratings_train.txt `
  --validation-ratio 0.1 `
  --output-dir Transformer/kcelectra_nsmc_model `
  --epochs 3 `
  --batch-size 8
```

처음에는 전체 학습 전에 작은 샘플로 실행 확인을 하는 것을 추천합니다.

```powershell
python Transformer/kcelectra_nsmc.py train `
  --train-sample 1000 `
  --validation-ratio 0.1 `
  --valid-sample 100 `
  --epochs 1
```

학습을 시작하면 train/validation 크기가 출력됩니다.

```text
train_size=135000 valid_size=15000
```

학습 중에는 각 epoch마다 validation 지표가 출력됩니다.

```text
epoch=1 valid_loss=0.4321 valid_accuracy=0.8123 valid_precision=0.8050 valid_recall=0.8260 valid_f1=0.8154
```

가장 낮은 `valid_loss`를 기록한 모델이 `Transformer/kcelectra_nsmc_model`에 저장됩니다.

## 별도 Validation 파일 사용

직접 만든 validation 파일이 있다면 `--valid-file`로 지정할 수 있습니다.
이 경우 `ratings_train.txt`를 자동 split하지 않습니다.

```powershell
python Transformer/kcelectra_nsmc.py train `
  --train-file Data/NSMC/ratings_train.txt `
  --valid-file Data/NSMC/ratings_valid.txt `
  --epochs 3
```

## 최종 Test 평가

저장된 모델을 `ratings_test.txt`로 최종 평가합니다.

```powershell
python Transformer/kcelectra_nsmc.py eval `
  --model-dir Transformer/kcelectra_nsmc_model `
  --test-file Data/NSMC/ratings_test.txt
```

출력 예시는 아래와 같습니다.

```text
Loss: 0.4213
Accuracy: 0.8234
Precision: 0.8175
Recall: 0.8341
F1: 0.8257
```

평가 지표 의미는 다음과 같습니다.

- `Loss`: 모델 예측과 정답 사이의 손실값입니다. 낮을수록 좋습니다.
- `Accuracy`: 전체 데이터 중 정답을 맞힌 비율입니다.
- `Precision`: 모델이 긍정이라고 예측한 것 중 실제 긍정인 비율입니다.
- `Recall`: 실제 긍정 데이터 중 모델이 긍정으로 맞힌 비율입니다.
- `F1`: Precision과 Recall의 조화평균입니다.

현재 `Precision`, `Recall`, `F1`은 positive label인 `1`을 기준으로 계산합니다.

## 예측

문장 하나를 입력해 긍정/부정 확률을 확인합니다.

```powershell
python Transformer/kcelectra_nsmc.py predict `
  --model-dir Transformer/kcelectra_nsmc_model `
  --text "정말 재미있는 영화였어요"
```

출력 예시는 JSON 형식입니다.

```json
{
  "text": "정말 재미있는 영화였어요",
  "label": "positive",
  "score": 0.9812,
  "negative": 0.0188,
  "positive": 0.9812
}
```

여러 문장을 파일로 입력하려면 `--input-file`을 사용할 수 있습니다.

```powershell
python Transformer/kcelectra_nsmc.py predict `
  --model-dir Transformer/kcelectra_nsmc_model `
  --input-file samples.txt
```

## 주요 옵션

- `--model-name`: 사용할 Hugging Face 모델 이름입니다. 기본값은 `beomi/KcELECTRA-base`입니다.
- `--train-file`: 학습에 사용할 NSMC 파일입니다.
- `--validation-ratio`: train 파일에서 validation으로 나눌 비율입니다. 기본값은 `0.1`입니다.
- `--valid-file`: 별도 validation 파일을 직접 지정합니다.
- `--test-file`: `eval` 명령에서 최종 평가에 사용할 test 파일입니다.
- `--max-length`: tokenizer가 사용할 최대 토큰 길이입니다. 기본값은 `128`입니다.
- `--device`: `auto`, `cpu`, `cuda` 중 선택합니다. 기본값은 `auto`입니다.
- `--train-sample`: 학습 데이터 일부만 샘플링해서 사용합니다.
- `--valid-sample`: validation 데이터 일부만 샘플링해서 사용합니다.
- `--eval-sample`: test 평가 데이터 일부만 샘플링해서 사용합니다.
- `--batch-size`: 학습 batch size입니다.
- `--eval-batch-size`: validation/test 평가 batch size입니다.

## 저장되는 파일

학습이 끝나면 기본적으로 아래 폴더에 모델이 저장됩니다.

```text
Transformer/kcelectra_nsmc_model/
```

저장 폴더에는 모델 가중치, config, tokenizer 파일, `training_args.json`,
`metrics.json`이 포함됩니다. 이후 `eval`, `predict`는 이 폴더에서
모델과 tokenizer를 함께 불러옵니다.
