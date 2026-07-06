# 프로젝트 계획서

## XAI를 활용한 딥러닝 모델의 의사결정 과정 분석

## 1. 프로젝트 개요

본 프로젝트는 한국어 영화 리뷰 감정 분석 데이터셋(NSMC)을 활용하여 FNN,
CNN, Transformer 기반 텍스트 분류 모델을 학습시키고, 각 모델이 특정
예측을 내릴 때 어떤 단어와 표현에 의존하는지 분석하는 것을 목표로 한다.

단순히 모델의 정확도를 비교하는 것이 아니라, Integrated Gradients,
Occlusion, Attention Visualization 등의 XAI 기법을 활용하여 모델별
의사결정 방식의 차이를 해석한다.

또한 심화 실험으로 Transformer 내부 hidden representation을 분석하여
긍정/부정을 구분하는 Sentiment Direction이 존재하는지 탐색한다. 이는
최근 AI Safety 분야에서 연구되는 Refusal Direction, Safety Direction,
Mechanistic Interpretability 연구를 이해하기 위한 기초적인 실험으로 볼
수 있다.

------------------------------------------------------------------------

# 2. 프로젝트 핵심 질문

1.  FNN, CNN, Transformer는 같은 문장을 보고도 서로 다른 판단 근거를
    사용하는가?
2.  각 모델은 어떤 단어와 표현에 의존하여 긍정/부정을 예측하는가?
3.  XAI 기법을 통해 모델의 판단 근거를 사람이 이해할 수 있는 형태로
    설명할 수 있는가?
4.  Transformer의 Attention Map은 Integrated Gradients나 Occlusion
    결과와 얼마나 일치하는가?
5.  Transformer 내부 표현 공간에는 긍정/부정을 구분하는 Sentiment
    Direction이 존재하는가?
6.  이러한 모델 해석 연구가 향후 AI Safety / Security 연구로 어떻게
    확장될 수 있는가?

------------------------------------------------------------------------

# 3. 사용할 데이터셋

## NSMC (Naver Sentiment Movie Corpus)

-   한국어 영화 리뷰 감정 분석 데이터셋
-   긍정 / 부정 이진 분류
-   label 0: 부정
-   label 1: 긍정

사용 링크

-   https://github.com/e9t/nsmc
-   https://huggingface.co/datasets/e9t/nsmc

------------------------------------------------------------------------

# 4. 전체 프로젝트 흐름

1.  NSMC 데이터 로드
2.  공통 전처리
3.  KoELECTRA 또는 KoBERT Tokenizer 적용
4.  FNN, CNN, Transformer 모델 학습
5.  모델 성능 비교
6.  Integrated Gradients 분석
7.  Occlusion 분석
8.  Transformer Attention Visualization
9.  모델별 의사결정 방식 비교
10. 심화 실험: Sentiment Direction 탐색

------------------------------------------------------------------------

# 5. 전처리 전략

원문을 최대한 유지하는 것을 원칙으로 한다.

## 적용

-   중복 데이터 제거
-   빈 문장 제거
-   Label 오류 확인
-   Unicode 정규화(NFC)
-   반복 공백 정리
-   최대 길이 초과 문장 Truncate

## 적용하지 않음

-   특수문자 제거
-   숫자 제거
-   영어 제거
-   불용어 제거
-   조사 제거
-   형태소 분석
-   ㅋㅋ, ㅠㅠ 제거

특수문자, 반복 표현, 별점, 이모티콘 등은 감정 정보가 될 수 있으므로
제거하지 않는다.

------------------------------------------------------------------------

# 6. Tokenizer 선택

FNN, CNN, Transformer 모두 동일한 WordPiece 기반 Tokenizer(KoELECTRA
또는 KoBERT)를 사용한다.

예)

-   배우
-   ##는
-   좋
-   ##았
-   ##지만
-   스토리
-   ##는
-   너무
-   지루
-   ##했다

XAI 결과 시각화 시에는 subword를 다시 하나의 단어로 합쳐 표시한다.

------------------------------------------------------------------------

# 7. 모델 구성

## FNN

Embedding → Mean Pooling → FC → ReLU → Dropout → Output

특징

-   단어 자체에 크게 의존
-   문맥 정보 활용이 어려움

------------------------------------------------------------------------

## CNN (TextCNN)

Embedding → Conv1D → ReLU → MaxPooling → FC → Output

특징

-   n-gram 패턴 학습
-   짧은 문맥 활용

------------------------------------------------------------------------

## Transformer

Tokenizer → Embedding → Transformer Encoder → \[CLS\] → Classifier

특징

-   긴 문맥 활용
-   Self-Attention 기반

------------------------------------------------------------------------

# 8. 실험 1 : 모델 성능 비교

평가 지표

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

세 모델을 동일한 Train/Test Split으로 학습하여 비교한다.

------------------------------------------------------------------------

# 9. 실험 2 : Integrated Gradients

목적

각 모델이 어떤 단어 때문에 긍정/부정을 판단했는지 분석한다.

비교

-   FNN : 감정 단어 중심
-   CNN : 감정 구절 중심
-   Transformer : 문맥 기반 판단

------------------------------------------------------------------------

# 10. 실험 3 : Occlusion

단어를 하나씩 제거하여 예측 확률 변화를 측정한다.

목적

Integrated Gradients 결과가 실제 예측 변화와 연결되는지 검증한다.

------------------------------------------------------------------------

# 11. 실험 4 : Attention Visualization

Transformer만 수행한다.

분석

-   어떤 단어끼리 연결되는가?
-   어떤 단어를 가장 많이 참고하는가?
-   Attention과 IG 결과는 얼마나 일치하는가?

------------------------------------------------------------------------

# 12. 실험 5 : 모델별 의사결정 비교

대표 문장을 선정하여

-   FNN
-   CNN
-   Transformer

세 모델의 중요 단어와 판단 근거를 비교한다.

예상

-   FNN : 감정 단어
-   CNN : 감정 구절
-   Transformer : 문맥 및 대조 관계

------------------------------------------------------------------------

# 13. 실험 6 : 특수문자 및 인터넷 표현 분석

예시

-   최고!!!!
-   시간 아까움;;
-   ㅋㅋㅋㅋ
-   ㅠㅠ
-   ★★★★★

질문

-   특수문자도 감정 판단에 영향을 주는가?
-   인터넷 표현도 중요한 Feature가 되는가?

------------------------------------------------------------------------

# 14. 심화 실험 : Sentiment Direction

Transformer Hidden State를 추출하여

v_sentiment = mean(h_positive) - mean(h_negative)

를 계산한다.

## 검증

1.  Projection Score
2.  Linear Probe
3.  XAI 결과와 비교

XAI에서 중요하게 나온 단어를 제거했을 때 Projection Score도 함께
변화하는지 분석한다.

------------------------------------------------------------------------

# 15. AI Safety / Mechanistic Interpretability와의 연결

본 프로젝트는 AI Safety 자체를 다루는 것이 아니라 모델을 이해하는 것을
목표로 한다.

최근 AI Safety 연구에서는 Refusal Direction, Safety Direction, Deception
Direction 등을 찾아 모델 내부를 이해하고 위험 행동을 탐지하거나
제어하려는 연구가 활발히 진행되고 있다.

본 프로젝트의 Sentiment Direction 실험은 이러한 연구의 기초적인
Representation Analysis 실험으로 볼 수 있다.

------------------------------------------------------------------------

# 16. 역할 분담

### 4인 팀

1. 모델 만들때

* FNN - 김태령
* CNN - 태현준
* Transformer - 최동근, 이승민

2. Xai기법

* FNN - 최동근
* CNN - 이승민
* Transformer - 김태령, 태현준

- 심화 실험은 이후 다같이
------------------------------------------------------------------------

# 17. 예상 결과

-   FNN은 감정 단어에 크게 의존할 가능성이 높다.
-   CNN은 감정 구절에 민감할 가능성이 높다.
-   Transformer는 문맥 관계를 활용할 가능성이 높다.
-   Attention과 IG는 완전히 일치하지 않을 수 있다.
-   특수문자와 인터넷 표현도 감정 판단에 활용될 수 있다.
-   Transformer 내부에 긍정/부정을 나타내는 방향이 존재할 가능성을
    탐색한다.

------------------------------------------------------------------------

# 18. 프로젝트의 의의

본 프로젝트는 단순히 감정 분석 성능을 비교하는 것이 아니라, 모델이 왜
그러한 판단을 내렸는지 해석하는 것을 목표로 한다.

이를 통해 모델 구조별 특징, Explainable AI(XAI), 모델 신뢰성, 그리고
향후 Mechanistic Interpretability 및 AI Safety 연구로 확장될 수 있는
기반을 이해하고자 한다.
