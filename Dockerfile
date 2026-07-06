# 1. 베이스 이미지 선택 (팀 가상환경 스펙 Python 3.11에 맞춤)
FROM python:3.11-slim

# 2. 필수 시스템 패키지 설치
# - build-essential, git: 빌드용 기본 도구
# - default-jdk: KoNLPy(Okt) 형태소 분석기 구동을 위한 자바 가상머신(JVM) 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# 3. 자바 환경변수 설정 (KoNLPy가 JVM 라이브러리를 정상 인식할 수 있도록 고정)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# 4. 작업 디렉터리 생성 및 설정
WORKDIR /workspace

# 5. 의존성 파일 복사 및 설치
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드 전체 복사
COPY . /workspace

# 7. 기본 명령어 (컨테이너 실행 시 bash 쉘로 진입)
CMD ["/bin/bash"]
