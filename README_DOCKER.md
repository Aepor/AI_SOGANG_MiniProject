# 🐳 XAI 프로젝트 도커(Docker) 개발 환경 안내서

이 안내서는 팀원들이 동일한 환경(Python 3.11, CUDA 12.6, PyTorch 등)에서 버그 없이 감정 분석 및 XAI 코드를 실행할 수 있도록 도커 환경을 설정하고 사용하는 방법을 설명합니다.

---

## 🛠️ 사전 준비 (최초 1회만 수행)

도커 내부에서 GPU(그래픽카드) 가속을 사용하려면 호스트 PC에 다음이 설치되어 있어야 합니다.

1.  **NVIDIA 그래픽카드 드라이버** 설치
2.  **Docker Desktop** 설치 (Windows/macOS)
3.  **NVIDIA Container Toolkit** 설치 (Windows는 WSL2 기반 Docker Desktop 설치 시 자동으로 포함되나, Linux의 경우 별도 설치 필요)

---

## 🚀 도커 컨테이너 빌드 및 실행

프로젝트 루트 디렉터리(`AI_SOGANG_MiniProject`)에서 터미널을 열고 아래 명령어를 수행합니다.

### 1. 이미지 빌드 및 컨테이너 실행 (docker-compose 사용)

`docker-compose.yml`이 이미 작성되어 있어 명령어 한 줄로 편리하게 켜고 끌 수 있습니다.

```bash
# 1. 도커 이미지 빌드 및 백그라운드 실행
docker compose up -d

# 2. 실행 중인 도커 컨테이너 내부 터미널(bash)로 접속
docker compose exec xai-app bash
```

> **💡 볼륨 마운트 안내:**
> `docker-compose.yml` 설정을 통해 로컬 컴퓨터의 파일들과 컨테이너 내부의 `/workspace` 폴더가 연동되어 있습니다. 
> 로컬에서 VS Code나 메모장으로 코드를 수정하면, 컨테이너 내부에 즉시 반영되므로 **코드를 수정할 때마다 도커 이미지를 새로 빌드할 필요가 없습니다.**

---

### 2. 컨테이너 내부에서 코드 실행하기

위 2번 명령어를 통해 컨테이너 내부(`root@...:/workspace#`)로 진입했다면, 로컬과 동일하게 명령어를 치고 실행하시면 됩니다.

**[예시: 1차 구동 (감정 방향성 도출)]**
```bash
python XAI/SentimentDirection/sentiment_direction.py \
  --model-dir XAI/Transformer/kcelectra_nsmc_model \
  --train-file Data/NSMC/ratings_train.txt \
  --test-file Data/NSMC/ratings_test.txt \
  --output-dir XAI/outputs_sentiment_direction
```

**[예시: 2차 구동 (XAI 어블레이션 분석)]**
```bash
python XAI/SentimentDirection/sentiment_direction_ablation.py \
  --model-dir XAI/Transformer/kcelectra_nsmc_model \
  --direction-dir XAI/outputs_sentiment_direction \
  --input-file inputs.txt \
  --output-dir XAI/outputs_sentiment_direction
```

---

## ⏹️ 컨테이너 종료 및 정리

작업을 모두 마친 뒤 컨테이너를 끄고 리소스를 반환하려면 호스트 PC 터미널에서 다음 명령어를 실행합니다.

```bash
# 컨테이너 중지 및 리소스 삭제
docker compose down
```
