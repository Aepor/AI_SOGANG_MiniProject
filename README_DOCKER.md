# 🐳 XAI 프로젝트 도커(Docker) 초보자 가이드 및 설명서

이 설명서는 도커(Docker)를 생전 처음 접하는 팀원들도 어려움 없이 개발 환경을 세팅하고 코드를 실행할 수 있도록 단계별로 안내합니다. 

도커를 사용하면 팀원 모두가 동일한 **Python 3.11, CUDA 12.6, PyTorch, 자바(JDK)** 개발 환경에서 에러 없이 코드를 실행할 수 있습니다.

---

## 💡 아주 쉬운 도커 개념 이해하기
*   **가상 컨테이너(컨테이너):** 내 원래 컴퓨터 환경을 해치지 않고, 그 위에 올려두는 **"조그맣고 깨끗한 인공지능용 미니 리눅스 컴퓨터"**입니다.
*   **실시간 공유 폴더 (볼륨 마운트):** 내 원래 컴퓨터의 프로젝트 폴더(`AI_SOGANG_MiniProject`)와 도커 내부 컴퓨터가 실시간으로 동기화됩니다. **코드는 원래대로 내 윈도우 편집기(VS Code 등)로 수정하고, 실행만 도커 터미널에서 수행**하면 됩니다.

---

## 🛠️ 1단계: 사전 준비 (최초 1회만 수행)

1.  **NVIDIA 그래픽카드 드라이버**가 최신으로 설치되어 있는지 확인합니다.
2.  **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** 공식 홈페이지에서 프로그램을 다운로드하여 설치합니다.
    *   *Windows 사용자는 설치 도중 나오는 **WSL 2** 권장 옵션을 반드시 켠 채로 설치해 주세요.*
3.  설치가 완료되면 **Docker Desktop 프로그램을 실행**해 둡니다. (오른쪽 아래 작업 표시줄에 도커 고래 아이콘이 조용히 떠 있으면 됩니다.)

---

## 🚀 2단계: 터미널 열고 도커 버전 확인하기

1.  윈도우 검색창에 `cmd` 혹은 `PowerShell`을 입력하여 **터미널을 실행**합니다.
2.  도커 엔진이 정상적으로 내 컴퓨터에서 켜져 있는지 확인하기 위해 아래 명령어를 입력합니다.
    ```bash
    docker --version
    ```
    *   `Docker version 24.x.x` 형태로 버전 정보가 정상 출력되면 준비 완료입니다.

---

## 📂 3단계: 프로젝트 폴더로 이동하여 가상 컨테이너 가동하기

1.  `cd` 명령어를 사용하여 터미널의 현재 위치를 **프로젝트 폴더**로 이동시킵니다.
    ```bash
    # 본인의 실제 프로젝트 폴더 경로를 입력합니다. (예시)
    cd C:\Users\user\Desktop\AI_SOGANG_MiniProject
    ```
2.  가상 컴퓨터를 조립하고 구동하기 위해 아래 명령어를 입력합니다.
    ```bash
    docker compose up -d
    ```
    *   **💡 참고:** 최초 실행 시에는 리눅스 운영체제 다운로드, 자바(JDK) 설치, PyTorch 등 대용량 패키지 다운로드 때문에 **몇 분 정도 소요**됩니다. (두 번째 실행할 때부터는 1초 만에 켜집니다.)

---

## 🖥️ 4단계: 가상 컨테이너 접속 및 코드 실행하기

1.  컨테이너 구동이 끝났다면, 아래 명령어를 쳐서 **가상 컴퓨터 터미널 내부로 접속**합니다.
    ```bash
    docker compose exec xai-app bash
    ```
    *   접속에 성공하면 터미널 앞부분이 `root@xxxxxx:/workspace#` 형태로 바뀝니다.
2.  이제 가상 리눅스 환경 안으로 들어왔습니다! **기존 로컬 컴퓨터에서 명령어를 쳤던 양식 그대로** 파이썬 코드를 실행할 수 있습니다.

    **[실행 예시 1: sentiment 분석 실행]**
    ```bash
    python XAI/SentimentDirection/sentiment_direction.py \
      --model-dir XAI/Transformer/kcelectra_nsmc_model \
      --train-file Data/NSMC/ratings_train.txt \
      --test-file Data/NSMC/ratings_test.txt \
      --output-dir XAI/outputs_sentiment_direction \
      --max-samples 10000
    ```

    **[실행 예시 2: CNN XAI 실행]**
    ```bash
    python XAI/CNN/nsmc_cnn_xai.py
    ```

---

## ⏹️ 5단계: 종료하기

작업을 모두 마쳤다면 도커 가상 컴퓨터를 안전하게 종료합니다.

1.  도커 터미널에서 빠져나오기 (윈도우 터미널로 복귀):
    ```bash
    exit
    ```
2.  가상 컴퓨터 전원 끄기 (리소스 반환):
    ```bash
    docker compose down
    ```

---

## ⚡ 팁: 수정 및 패키지 추가 시 빠른 업데이트 방법 (캐싱 기능)

### 1. 코드 파일(`.py` 등)을 수정한 경우
*   `docker-compose.yml` 설정 덕분에 로컬 컴퓨터에서 코드를 수정하면 도커 내부에 **실시간으로 즉시 반영**됩니다. 따로 도커를 껐다 켜거나 재빌드할 필요가 전혀 없습니다.

### 2. `requirements.txt`에 새 패키지를 추가한 경우
*   기존의 무거운 패키지(PyTorch 등)를 처음부터 다시 다운로드하지 않고, **새로 추가된 패키지만 몇 초 만에 빠르게 다운로드**하여 업데이트합니다. (도커의 레이어 캐싱 기능 덕분)
*   패키지를 추가한 뒤, 아래 명령어를 실행하면 변경된 부분만 빠르게 조립하여 가동됩니다.
    ```bash
    docker compose up -d --build
    ```
