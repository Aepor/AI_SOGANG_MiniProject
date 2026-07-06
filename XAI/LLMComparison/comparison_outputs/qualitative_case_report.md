# LLM-XAI Qualitative Case Report

## case_001

text: 이 영화 정말 재미있어요 추천합니다

- LLM sentiment: positive
- LLM evidence: 정말 재미있어요, 추천합니다
- LLM reason: 재미있다고 평가하고 추천한다는 표현이 있어 긍정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | filter_activation | positive | 정말 | 재미있어요 | 추천합니다 | 0.999423 |
| transformer | integrated_gradients_steps100 | positive | 정말 | 재미있어요 | 추천합니다 | 0.982569 |
| transformer | integrated_gradients_steps50 | positive | 정말 | 재미있어요 | 추천합니다 | 0.977997 |
| cnn | ngram_occlusion | positive | 정말 | 재미있어요 | 추천합니다 | 0.973118 |
| fnn | integrated_gradients_steps50 | positive | 영화 | 재미있어요 | 추천합니다 | 0.694362 |
| fnn | integrated_gradients_steps100 | positive | 영화 | 재미있어요 | 추천합니다 | 0.693115 |
| cnn | integrated_gradients_steps50 | positive | 영화 | 정말 | 재미있어요 | 0.660879 |
| cnn | integrated_gradients_steps100 | positive | 영화 | 정말 | 재미있어요 | 0.660639 |
| fnn | lime | positive | 영화 | 재미있어요 | 추천합니다 | 0.660489 |
| cnn | unigram_occlusion | positive | 영화 | 정말 | 재미있어요 | 0.659106 |
| fnn | occlusion | positive | 영화 | 재미있어요 | 추천합니다 | 0.656038 |
| transformer | occlusion | positive | 영화 | 재미있어요 | 추천합니다 | 0.646183 |

## case_002

text: 스토리가 지루하고 배우들 연기도 별로네요

- LLM sentiment: negative
- LLM evidence: 스토리가 지루하고, 배우들 연기도 별로네요
- LLM reason: 스토리가 지루하고 배우들 연기도 별로라고 평가한 명확한 부정 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | ngram_occlusion | negative | 배우들 | 연기도 | 별로네요 | 0.722685 |
| cnn | filter_activation | negative | 지루하고 | 배우들 | 별로네요 | 0.71306 |
| fnn | lime | negative | 지루하고 | 배우들 | 별로네요 | 0.697209 |
| fnn | integrated_gradients_steps100 | negative | 지루하고 | 배우들 | 별로네요 | 0.675723 |
| cnn | integrated_gradients_steps100 | negative | 스토리가 | 지루하고 | 별로네요 | 0.675274 |
| cnn | integrated_gradients_steps50 | negative | 스토리가 | 지루하고 | 별로네요 | 0.675205 |
| fnn | integrated_gradients_steps50 | negative | 지루하고 | 배우들 | 별로네요 | 0.674963 |
| transformer | integrated_gradients_steps100 | negative | 지루하고 | 배우들 | 별로네요 | 0.665009 |
| fnn | occlusion | negative | 스토리가 | 지루하고 | 별로네요 | 0.64451 |
| transformer | occlusion | negative | 배우들 | 연기도 | 별로네요 | 0.624481 |
| cnn | unigram_occlusion | negative | 지루하고 | 연기도 | 별로네요 | 0.620326 |
| transformer | integrated_gradients_steps50 | negative | 지루하고 | 배우들 | 연기도 | 0.564622 |

## case_003

text: 처음엔 지루했는데 갈수록 정말 재미있어지네요

- LLM sentiment: positive
- LLM evidence: 처음엔 지루했는데, 갈수록 정말 재미있어지네요
- LLM reason: 초반의 지루함은 언급되지만, 갈수록 정말 재미있어졌다는 최종 평가가 더 강해 전체적으로 긍정입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | ngram_occlusion | positive | 갈수록 | 정말 | 재미있어지네요 | 0.7358 |
| transformer | integrated_gradients_steps50 | positive | 갈수록 | 정말 | 재미있어지네요 | 0.680595 |
| cnn | unigram_occlusion | positive | 지루했는데 | 정말 | 재미있어지네요 | 0.679984 |
| transformer | integrated_gradients_steps100 | positive | 갈수록 | 정말 | 재미있어지네요 | 0.672312 |
| cnn | integrated_gradients_steps50 | positive | 지루했는데 | 정말 | 재미있어지네요 | 0.672131 |
| cnn | integrated_gradients_steps100 | positive | 지루했는데 | 정말 | 재미있어지네요 | 0.671919 |
| fnn | lime | positive | 지루했는데 | 정말 | 재미있어지네요 | 0.661173 |
| transformer | occlusion | positive | 지루했는데 | 갈수록 | 재미있어지네요 | 0.645757 |
| fnn | occlusion | positive | 지루했는데 | 갈수록 | 재미있어지네요 | 0.639858 |
| fnn | integrated_gradients_steps50 | positive | 처음엔 | 지루했는데 | 재미있어지네요 | 0.612185 |
| fnn | integrated_gradients_steps100 | positive | 처음엔 | 지루했는데 | 재미있어지네요 | 0.611585 |
| cnn | filter_activation | positive | 처음엔 | 갈수록 | 정말 | 0.51 |

## case_004

text: 배우들의 연기는 명품인데 각본이 너무 쓰레기네요

- LLM sentiment: negative
- LLM evidence: 배우들의 연기는 명품인데, 각본이 너무 쓰레기네요
- LLM reason: 연기는 칭찬하지만, 핵심 요소인 각본을 '너무 쓰레기'라고 강하게 비판해 전체적으로 부정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | integrated_gradients_steps100 | negative | 명품인데 | 각본이 | 쓰레기네요 | 0.63483 |
| cnn | integrated_gradients_steps50 | negative | 명품인데 | 각본이 | 쓰레기네요 | 0.634526 |
| cnn | filter_activation | negative | 명품인데 | 너무 | 쓰레기네요 | 0.63145 |
| cnn | ngram_occlusion | negative | 각본이 | 너무 | 쓰레기네요 | 0.629816 |
| transformer | integrated_gradients_steps100 | negative | 각본이 | 너무 | 쓰레기네요 | 0.622635 |
| fnn | lime | negative | 명품인데 | 각본이 | 쓰레기네요 | 0.616404 |
| fnn | integrated_gradients_steps100 | negative | 연기는 | 명품인데 | 쓰레기네요 | 0.607833 |
| fnn | integrated_gradients_steps50 | negative | 연기는 | 명품인데 | 쓰레기네요 | 0.607765 |
| fnn | occlusion | negative | 명품인데 | 각본이 | 쓰레기네요 | 0.599308 |
| cnn | unigram_occlusion | negative | 각본이 | 너무 | 쓰레기네요 | 0.588227 |
| transformer | integrated_gradients_steps50 | negative | 명품인데 | 너무 | 쓰레기네요 | 0.545298 |
| transformer | occlusion | negative | 연기는 | 너무 | 쓰레기네요 | 0.52738 |

## case_005

text: 와... 이런 영화에 내 피 같은 돈을 쓰다니 대단하다

- LLM sentiment: negative
- LLM evidence: 내 피 같은 돈을 쓰다니, 이런 영화에, 대단하다
- LLM reason: 소중한 돈을 쓴 것을 후회하며 영화에 대한 불만을 반어적으로 표현한 부정 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | ngram_occlusion | negative | 피 | 같은 | 돈을 | 0.53624 |
| cnn | filter_activation | negative | 돈을 | 쓰다니 | 대단하다 | 0.5268 |
| fnn | occlusion | negative | 피 | 돈을 | 대단하다 | 0.493305 |
| cnn | integrated_gradients_steps100 | negative | 돈을 | 쓰다니 | 대단하다 | 0.477446 |
| cnn | integrated_gradients_steps50 | negative | 돈을 | 쓰다니 | 대단하다 | 0.477251 |
| transformer | occlusion | negative | 영화에 | 돈을 | 대단하다 | 0.472347 |
| transformer | integrated_gradients_steps100 | negative | 이런 | 영화에 | 돈을 | 0.441985 |
| cnn | unigram_occlusion | negative | 같은 | 돈을 | 대단하다 | 0.420849 |
| transformer | integrated_gradients_steps50 | negative | 이런 | 돈을 | 대단하다 | 0.410752 |
| fnn | lime | negative | 와... | 돈을 | 대단하다 | 0.372141 |
| fnn | integrated_gradients_steps100 | negative | 와... | 돈을 | 대단하다 | 0.320678 |
| fnn | integrated_gradients_steps50 | negative | 와... | 돈을 | 대단하다 | 0.320532 |

## case_006

text: 수면제 대신 보면 딱 좋은 훌륭한 영화입니다

- LLM sentiment: negative
- LLM evidence: 수면제 대신 보면 딱 좋은
- LLM reason: 겉으로는 좋은·훌륭한 영화라고 하지만, 수면제 대신 볼 만하다는 표현이 지루함을 비꼬는 말이라 부정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| fnn | lime | positive | 수면제 | 좋은 | 훌륭한 | 0.401038 |
| cnn | integrated_gradients_steps100 | positive | 수면제 | 좋은 | 훌륭한 | 0.364013 |
| cnn | integrated_gradients_steps50 | positive | 수면제 | 좋은 | 훌륭한 | 0.363739 |
| fnn | integrated_gradients_steps50 | positive | 수면제 | 좋은 | 훌륭한 | 0.361101 |
| fnn | integrated_gradients_steps100 | positive | 수면제 | 좋은 | 훌륭한 | 0.361031 |
| cnn | unigram_occlusion | positive | 보면 | 좋은 | 훌륭한 | 0.306112 |
| cnn | filter_activation | positive | 수면제 | 훌륭한 | 영화입니다 | 0.287375 |
| transformer | occlusion | positive | 딱 | 훌륭한 | 영화입니다 | 0.234027 |
| fnn | occlusion | positive | 좋은 | 훌륭한 | 영화입니다 | 0.183547 |
| transformer | integrated_gradients_steps100 | positive | 딱 | 훌륭한 | 영화입니다 | 0.181195 |
| cnn | ngram_occlusion | positive | 좋은 | 훌륭한 | 영화입니다 | 0.178807 |
| transformer | integrated_gradients_steps50 | positive | 딱 | 훌륭한 | 영화입니다 | 0.173658 |

## case_007

text: 와 진짜 개존잼!!! 꼭보세요 ㅋㅋㅋ

- LLM sentiment: positive
- LLM evidence: 진짜 개존잼!!!, 꼭보세요, ㅋㅋㅋ
- LLM reason: 매우 재미있다고 강조하고 관람을 추천하는 강한 긍정 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps50 | positive | 개존잼!!! | 꼭보세요 | ㅋㅋㅋ | 0.824428 |
| transformer | integrated_gradients_steps100 | positive | 개존잼!!! | 꼭보세요 | ㅋㅋㅋ | 0.82229 |
| cnn | filter_activation | positive | 개존잼!!! | 꼭보세요 | ㅋㅋㅋ | 0.80499 |
| cnn | ngram_occlusion | positive | 개존잼!!! | 꼭보세요 | ㅋㅋㅋ | 0.804488 |
| cnn | integrated_gradients_steps50 | positive | 진짜 | 개존잼!!! | 꼭보세요 | 0.768256 |
| cnn | integrated_gradients_steps100 | positive | 진짜 | 개존잼!!! | 꼭보세요 | 0.768243 |
| transformer | occlusion | positive | 진짜 | 개존잼!!! | 꼭보세요 | 0.766392 |
| cnn | unigram_occlusion | positive | 개존잼!!! | 꼭보세요 | ㅋㅋㅋ | 0.724834 |
| fnn | lime | positive | 와 | 진짜 | 꼭보세요 | 0.539019 |
| fnn | integrated_gradients_steps100 | positive | 와 | 진짜 | 꼭보세요 | 0.511519 |
| fnn | integrated_gradients_steps50 | positive | 와 | 진짜 | 꼭보세요 | 0.51074 |
| fnn | occlusion | positive | 와 | 개존잼!!! | 꼭보세요 | 0.481077 |

## case_008

text: 시간낭비 오지구요 지리구요 다신 안봄 ㅡㅡ

- LLM sentiment: negative
- LLM evidence: 시간낭비, 다신 안봄, ㅡㅡ
- LLM reason: 시간낭비였고 다시 보지 않겠다는 표현이 있어 부정적 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| fnn | lime | negative | 시간낭비 | 다신 | ㅡㅡ | 0.798702 |
| cnn | integrated_gradients_steps100 | negative | 시간낭비 | 다신 | 안봄 | 0.778709 |
| cnn | integrated_gradients_steps50 | negative | 시간낭비 | 다신 | 안봄 | 0.778372 |
| cnn | unigram_occlusion | negative | 시간낭비 | 다신 | 안봄 | 0.745718 |
| transformer | integrated_gradients_steps100 | negative | 다신 | 안봄 | ㅡㅡ | 0.740012 |
| transformer | occlusion | negative | 다신 | 안봄 | ㅡㅡ | 0.622872 |
| fnn | occlusion | negative | 시간낭비 | 지리구요 | ㅡㅡ | 0.570454 |
| fnn | integrated_gradients_steps50 | negative | 시간낭비 | 지리구요 | ㅡㅡ | 0.564578 |
| fnn | integrated_gradients_steps100 | negative | 시간낭비 | 지리구요 | ㅡㅡ | 0.564188 |
| transformer | integrated_gradients_steps50 | negative | 지리구요 | 다신 | 안봄 | 0.555844 |
| cnn | ngram_occlusion | negative | 시간낭비 | 오지구요 | 지리구요 | 0.387328 |
| cnn | filter_activation | negative | 시간낭비 | 오지구요 | 지리구요 | 0.36064 |

## case_009

text: 스토리가 지루하긴 한데 배우들은 멋있음

- LLM sentiment: positive
- LLM evidence: 스토리가 지루하긴, 배우들은 멋있음
- LLM reason: 부정적인 언급이 있지만, '한데' 뒤에서 배우들에 대한 긍정 평가가 최종적으로 제시되어 전체적으로는 약한 긍정으로 판단함.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps100 | positive | 스토리가 | 배우들은 | 멋있음 | 0.764471 |
| transformer | integrated_gradients_steps50 | positive | 스토리가 | 배우들은 | 멋있음 | 0.76274 |
| cnn | integrated_gradients_steps100 | positive | 지루하긴 | 배우들은 | 멋있음 | 0.711962 |
| cnn | integrated_gradients_steps50 | positive | 지루하긴 | 배우들은 | 멋있음 | 0.711755 |
| cnn | ngram_occlusion | positive | 한데 | 배우들은 | 멋있음 | 0.618915 |
| cnn | unigram_occlusion | positive | 지루하긴 | 한데 | 멋있음 | 0.568718 |
| transformer | occlusion | positive | 지루하긴 | 한데 | 멋있음 | 0.534372 |
| fnn | integrated_gradients_steps100 | negative | 지루하긴 | 한데 | 멋있음 | 0.506923 |
| fnn | integrated_gradients_steps50 | negative | 지루하긴 | 한데 | 멋있음 | 0.506517 |
| fnn | lime | negative | 지루하긴 | 한데 | 멋있음 | 0.487626 |
| fnn | occlusion | negative | 지루하긴 | 한데 | 멋있음 | 0.464262 |
| cnn | filter_activation | positive | 지루하긴 | 한데 | 배우들은 | 0.40068 |

## case_010

text: 상당히 재미있는 영화에요 친구들이나 부모님, 가족끼리 보면 더 재밌을 것 같아요

- LLM sentiment: positive
- LLM evidence: 상당히 재미있는 영화에요, 더 재밌을 것 같아요
- LLM reason: 영화를 재미있다고 평가하고 가족이나 친구들과 보면 더 즐거울 것이라고 말해 전반적으로 긍정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | integrated_gradients_steps100 | positive | 재미있는 | 영화에요 | 재밌을 | 0.486222 |
| cnn | integrated_gradients_steps50 | positive | 재미있는 | 영화에요 | 재밌을 | 0.486145 |
| cnn | ngram_occlusion | positive | 재미있는 | 영화에요 | 친구들이나 | 0.445136 |
| transformer | occlusion | positive | 재미있는 | 영화에요 | 부모님, | 0.440328 |
| cnn | unigram_occlusion | positive | 재미있는 | 부모님, | 재밌을 | 0.402214 |
| cnn | filter_activation | positive | 재미있는 | 영화에요 | 친구들이나 | 0.387869 |
| transformer | integrated_gradients_steps50 | positive | 재미있는 | 영화에요 | 부모님, | 0.360931 |
| transformer | integrated_gradients_steps100 | positive | 재미있는 | 영화에요 | 부모님, | 0.355382 |

## case_011

text: 평점 알바들만 넘치네 진짜 돈 아깝다

- LLM sentiment: negative
- LLM evidence: 평점 알바들만 넘치네, 진짜 돈 아깝다
- LLM reason: 평점 조작 의심과 돈이 아깝다는 표현으로 강한 부정 평가입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | ngram_occlusion | negative | 알바들만 | 돈 | 아깝다 | 0.670206 |
| transformer | occlusion | negative | 알바들만 | 돈 | 아깝다 | 0.620258 |
| cnn | unigram_occlusion | negative | 알바들만 | 돈 | 아깝다 | 0.617702 |
| cnn | filter_activation | negative | 알바들만 | 넘치네 | 진짜 | 0.616204 |
| fnn | lime | negative | 알바들만 | 돈 | 아깝다 | 0.60971 |
| fnn | occlusion | negative | 알바들만 | 돈 | 아깝다 | 0.594733 |
| fnn | integrated_gradients_steps50 | negative | 알바들만 | 돈 | 아깝다 | 0.576052 |
| cnn | integrated_gradients_steps100 | negative | 알바들만 | 돈 | 아깝다 | 0.575716 |
| fnn | integrated_gradients_steps100 | negative | 알바들만 | 돈 | 아깝다 | 0.575297 |
| cnn | integrated_gradients_steps50 | negative | 알바들만 | 돈 | 아깝다 | 0.575228 |
| transformer | integrated_gradients_steps100 | negative | 진짜 | 돈 | 아깝다 | 0.561918 |
| transformer | integrated_gradients_steps50 | negative | 진짜 | 돈 | 아깝다 | 0.53256 |

## case_012

text: 어설픈 신파보다 백배 천배 낫다. 가슴 먹먹해지는 수작.

- LLM sentiment: positive
- LLM evidence: 백배 천배 낫다., 가슴 먹먹해지는 수작.
- LLM reason: 어설픈 신파와 대비해 훨씬 낫고, 감정적 울림이 있는 수작이라고 평가한다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps100 | positive | 천배 | 가슴 | 수작. | 0.639576 |
| transformer | integrated_gradients_steps50 | positive | 천배 | 가슴 | 수작. | 0.636752 |
| cnn | ngram_occlusion | positive | 가슴 | 먹먹해지는 | 수작. | 0.622115 |
| cnn | unigram_occlusion | positive | 가슴 | 먹먹해지는 | 수작. | 0.609525 |
| cnn | integrated_gradients_steps50 | positive | 가슴 | 먹먹해지는 | 수작. | 0.578981 |
| cnn | integrated_gradients_steps100 | positive | 가슴 | 먹먹해지는 | 수작. | 0.57893 |
| cnn | filter_activation | positive | 신파보다 | 가슴 | 먹먹해지는 | 0.473098 |
| fnn | occlusion | positive | 어설픈 | 가슴 | 수작. | 0.468716 |
| transformer | occlusion | positive | 신파보다 | 먹먹해지는 | 수작. | 0.467328 |
| fnn | integrated_gradients_steps100 | positive | 어설픈 | 가슴 | 수작. | 0.41197 |
| fnn | integrated_gradients_steps50 | positive | 어설픈 | 가슴 | 수작. | 0.411732 |
| fnn | lime | positive | 어설픈 | 가슴 | 수작. | 0.404234 |

## case_013

text: 솔직히 재미는 없음... 시간 때우기용도 안됨

- LLM sentiment: negative
- LLM evidence: 재미는 없음..., 시간 때우기용도 안됨
- LLM reason: 재미가 없고 시간 때우기에도 부적합하다는 명확한 부정 평가입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| fnn | integrated_gradients_steps100 | negative | 없음... | 때우기용도 | 안됨 | 0.718253 |
| fnn | integrated_gradients_steps50 | negative | 없음... | 때우기용도 | 안됨 | 0.717993 |
| cnn | filter_activation | negative | 재미는 | 없음... | 시간 | 0.703552 |
| cnn | ngram_occlusion | negative | 없음... | 시간 | 때우기용도 | 0.700972 |
| cnn | integrated_gradients_steps50 | negative | 재미는 | 없음... | 때우기용도 | 0.698925 |
| cnn | integrated_gradients_steps100 | negative | 재미는 | 없음... | 때우기용도 | 0.698699 |
| transformer | integrated_gradients_steps50 | negative | 재미는 | 없음... | 때우기용도 | 0.698198 |
| transformer | integrated_gradients_steps100 | negative | 재미는 | 없음... | 때우기용도 | 0.696877 |
| fnn | lime | negative | 없음... | 때우기용도 | 안됨 | 0.692622 |
| fnn | occlusion | negative | 없음... | 때우기용도 | 안됨 | 0.683833 |
| cnn | unigram_occlusion | negative | 재미는 | 없음... | 때우기용도 | 0.664432 |
| transformer | occlusion | negative | 재미는 | 없음... | 때우기용도 | 0.507476 |

## case_014

text: 개연성이 밥말아먹었나 스토리가 왜 이 모양임?

- LLM sentiment: negative
- LLM evidence: 개연성이 밥말아먹었나, 스토리가 왜 이 모양임?
- LLM reason: 개연성과 스토리에 대한 비판과 불만이 명확해 부정적 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | filter_activation | negative | 개연성이 | 밥말아먹었나 | 스토리가 | 0.650908 |
| cnn | integrated_gradients_steps50 | negative | 개연성이 | 밥말아먹었나 | 모양임? | 0.644552 |
| cnn | integrated_gradients_steps100 | negative | 개연성이 | 밥말아먹었나 | 모양임? | 0.644536 |
| cnn | ngram_occlusion | negative | 개연성이 | 밥말아먹었나 | 스토리가 | 0.635298 |
| fnn | integrated_gradients_steps50 | negative | 개연성이 | 밥말아먹었나 | 스토리가 | 0.630661 |
| fnn | integrated_gradients_steps100 | negative | 개연성이 | 밥말아먹었나 | 스토리가 | 0.62915 |
| transformer | integrated_gradients_steps50 | negative | 밥말아먹었나 | 스토리가 | 모양임? | 0.62277 |
| fnn | lime | negative | 개연성이 | 밥말아먹었나 | 모양임? | 0.619121 |
| transformer | integrated_gradients_steps100 | negative | 밥말아먹었나 | 스토리가 | 모양임? | 0.606797 |
| fnn | occlusion | negative | 개연성이 | 밥말아먹었나 | 스토리가 | 0.591158 |
| cnn | unigram_occlusion | negative | 개연성이 | 밥말아먹었나 | 스토리가 | 0.581337 |
| transformer | occlusion | negative | 개연성이 | 스토리가 | 왜 | 0.542546 |

## case_015

text: 연출, 연기, 음악 삼박자가 완벽하다. 꼭 극장가서 보세요.

- LLM sentiment: positive
- LLM evidence: 연출, 연기, 음악 삼박자가 완벽하다., 꼭 극장가서 보세요.
- LLM reason: 영화의 주요 요소를 완벽하다고 칭찬하고 관람을 적극 추천하므로 긍정적 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps100 | positive | 완벽하다. | 극장가서 | 보세요. | 0.584949 |
| transformer | integrated_gradients_steps50 | positive | 완벽하다. | 극장가서 | 보세요. | 0.555714 |
| fnn | integrated_gradients_steps100 | positive | 음악 | 완벽하다. | 꼭 | 0.547763 |
| fnn | integrated_gradients_steps50 | positive | 음악 | 완벽하다. | 꼭 | 0.545123 |
| transformer | occlusion | positive | 완벽하다. | 꼭 | 보세요. | 0.533811 |
| cnn | filter_activation | positive | 음악 | 완벽하다. | 꼭 | 0.533718 |
| cnn | ngram_occlusion | positive | 삼박자가 | 완벽하다. | 꼭 | 0.527403 |
| fnn | occlusion | positive | 음악 | 완벽하다. | 꼭 | 0.505475 |
| fnn | lime | positive | 완벽하다. | 꼭 | 극장가서 | 0.499298 |
| cnn | unigram_occlusion | positive | 삼박자가 | 완벽하다. | 극장가서 | 0.456095 |
| cnn | integrated_gradients_steps100 | positive | 삼박자가 | 완벽하다. | 꼭 | 0.449418 |
| cnn | integrated_gradients_steps50 | positive | 삼박자가 | 완벽하다. | 꼭 | 0.449337 |

## case_016

text: 하도 재밌다길래 봤는데 진짜 개노잼임 절대보지마셈

- LLM sentiment: negative
- LLM evidence: 진짜 개노잼임, 절대보지마셈
- LLM reason: 기대와 달리 매우 재미없다고 평가하며 보지 말라고 하므로 부정적 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | ngram_occlusion | negative | 진짜 | 개노잼임 | 절대보지마셈 | 0.941351 |
| transformer | integrated_gradients_steps50 | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.665713 |
| cnn | filter_activation | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.659786 |
| transformer | occlusion | negative | 봤는데 | 개노잼임 | 절대보지마셈 | 0.654886 |
| cnn | unigram_occlusion | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.654876 |
| transformer | integrated_gradients_steps100 | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.648231 |
| fnn | occlusion | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.643033 |
| fnn | lime | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.604629 |
| cnn | integrated_gradients_steps50 | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.59513 |
| cnn | integrated_gradients_steps100 | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.595112 |
| fnn | integrated_gradients_steps50 | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.564209 |
| fnn | integrated_gradients_steps100 | negative | 재밌다길래 | 개노잼임 | 절대보지마셈 | 0.563865 |

## case_017

text: 오랜만에 가슴 따뜻해지는 웰메이드 영화 한 편 봤네요. 강추!!

- LLM sentiment: positive
- LLM evidence: 가슴 따뜻해지는, 웰메이드, 강추!!
- LLM reason: 따뜻한 감동, 높은 완성도, 강한 추천 표현이 모두 포함된 긍정 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | integrated_gradients_steps100 | positive | 가슴 | 따뜻해지는 | 웰메이드 | 0.817285 |
| cnn | integrated_gradients_steps50 | positive | 가슴 | 따뜻해지는 | 웰메이드 | 0.817235 |
| fnn | occlusion | positive | 따뜻해지는 | 웰메이드 | 강추!! | 0.765999 |
| cnn | filter_activation | positive | 가슴 | 따뜻해지는 | 웰메이드 | 0.754681 |
| cnn | ngram_occlusion | positive | 가슴 | 따뜻해지는 | 웰메이드 | 0.753784 |
| fnn | lime | positive | 오랜만에 | 따뜻해지는 | 웰메이드 | 0.592833 |
| transformer | integrated_gradients_steps100 | positive | 웰메이드 | 한 | 강추!! | 0.578387 |
| cnn | unigram_occlusion | positive | 오랜만에 | 따뜻해지는 | 웰메이드 | 0.575576 |
| fnn | integrated_gradients_steps50 | positive | 오랜만에 | 따뜻해지는 | 강추!! | 0.574062 |
| transformer | integrated_gradients_steps50 | positive | 웰메이드 | 한 | 강추!! | 0.570428 |
| fnn | integrated_gradients_steps100 | positive | 오랜만에 | 따뜻해지는 | 강추!! | 0.570334 |
| transformer | occlusion | positive | 한 | 봤네요. | 강추!! | 0.379965 |

## case_018

text: 뻔하디 뻔한 한국식 억지 감동 짜내기 영화 극혐

- LLM sentiment: negative
- LLM evidence: 뻔하디 뻔한, 억지 감동 짜내기, 극혐
- LLM reason: 진부함, 억지 감동, 강한 혐오 표현이 결합된 명확한 부정 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | filter_activation | negative | 억지 | 감동 | 짜내기 | 0.643067 |
| fnn | occlusion | negative | 뻔한 | 억지 | 극혐 | 0.637847 |
| transformer | integrated_gradients_steps100 | negative | 억지 | 짜내기 | 극혐 | 0.61961 |
| transformer | integrated_gradients_steps50 | negative | 억지 | 짜내기 | 극혐 | 0.61487 |
| fnn | lime | negative | 억지 | 감동 | 극혐 | 0.610703 |
| cnn | unigram_occlusion | negative | 억지 | 짜내기 | 극혐 | 0.599962 |
| fnn | integrated_gradients_steps50 | negative | 억지 | 감동 | 극혐 | 0.581036 |
| fnn | integrated_gradients_steps100 | negative | 억지 | 감동 | 극혐 | 0.5807 |
| cnn | integrated_gradients_steps50 | negative | 한국식 | 억지 | 극혐 | 0.455269 |
| cnn | integrated_gradients_steps100 | negative | 한국식 | 억지 | 극혐 | 0.45513 |
| cnn | ngram_occlusion | negative | 한국식 | 억지 | 짜내기 | 0.437326 |
| transformer | occlusion | negative | 한국식 | 억지 | 극혐 | 0.286589 |

## case_019

text: 내 돈이랑 시간 돌려내라 ㅋㅋㅋ 감독 누구냐 진짜

- LLM sentiment: negative
- LLM evidence: 돈이랑 시간 돌려내라, 감독 누구냐 진짜
- LLM reason: 돈과 시간을 돌려달라는 표현과 감독에 대한 비꼼이 있어 부정적 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | filter_activation | negative | 돈이랑 | 감독 | 누구냐 | 0.646411 |
| cnn | ngram_occlusion | negative | 돈이랑 | 감독 | 누구냐 | 0.63473 |
| fnn | lime | negative | 돈이랑 | 감독 | 누구냐 | 0.622782 |
| transformer | occlusion | negative | 돈이랑 | 돌려내라 | 감독 | 0.617157 |
| fnn | integrated_gradients_steps50 | negative | 돈이랑 | 감독 | 누구냐 | 0.596189 |
| fnn | integrated_gradients_steps100 | negative | 돈이랑 | 감독 | 누구냐 | 0.594024 |
| fnn | occlusion | negative | 돈이랑 | 감독 | 누구냐 | 0.59198 |
| cnn | unigram_occlusion | negative | 돈이랑 | 감독 | 누구냐 | 0.585667 |
| cnn | integrated_gradients_steps100 | negative | 돌려내라 | 감독 | 누구냐 | 0.572354 |
| cnn | integrated_gradients_steps50 | negative | 돌려내라 | 감독 | 누구냐 | 0.572254 |
| transformer | integrated_gradients_steps50 | negative | 돌려내라 | 감독 | 누구냐 | 0.530684 |
| transformer | integrated_gradients_steps100 | negative | 돌려내라 | 감독 | 누구냐 | 0.523018 |

## case_020

text: 배우들 얼굴 보러 갔다가 힐링 제대로 하고 옴 ㅠㅠ 연기도 찰떡

- LLM sentiment: positive
- LLM evidence: 힐링 제대로 하고 옴, 연기도 찰떡
- LLM reason: 영화 관람 후 힐링을 느꼈고 연기도 좋게 평가하고 있어 긍정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps100 | positive | 힐링 | 옴 | 찰떡 | 0.629516 |
| transformer | integrated_gradients_steps50 | positive | 힐링 | 제대로 | 찰떡 | 0.616843 |
| cnn | filter_activation | positive | 힐링 | 옴 | ㅠㅠ | 0.489732 |
| transformer | occlusion | positive | 힐링 | ㅠㅠ | 찰떡 | 0.461203 |
| cnn | ngram_occlusion | positive | 갔다가 | 힐링 | 제대로 | 0.443194 |
| fnn | occlusion | positive | 힐링 | ㅠㅠ | 연기도 | 0.397147 |
| cnn | integrated_gradients_steps100 | positive | 힐링 | 제대로 | ㅠㅠ | 0.3916 |
| cnn | integrated_gradients_steps50 | positive | 힐링 | 제대로 | ㅠㅠ | 0.391042 |
| fnn | integrated_gradients_steps50 | positive | 힐링 | ㅠㅠ | 연기도 | 0.367442 |
| fnn | integrated_gradients_steps100 | positive | 힐링 | ㅠㅠ | 연기도 | 0.367362 |
| cnn | unigram_occlusion | positive | 힐링 | ㅠㅠ | 찰떡 | 0.347121 |
| fnn | lime | positive | 배우들 | 힐링 | ㅠㅠ | 0.248117 |

## case_021

text: 초반엔 오? 하다가 후반 갈수록 산으로 가네 ㅋㅋㅋ 어이없음

- LLM sentiment: negative
- LLM evidence: 후반 갈수록 산으로 가네, 어이없음, 초반엔 오?
- LLM reason: 초반의 약한 호기심에도 불구하고 후반부 전개가 산으로 가고 어이없다는 평가가 핵심이므로 부정적이다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps50 | negative | 산으로 | 가네 | 어이없음 | 0.591758 |
| cnn | filter_activation | negative | 갈수록 | 산으로 | 가네 | 0.581279 |
| transformer | integrated_gradients_steps100 | negative | 산으로 | 가네 | 어이없음 | 0.572234 |
| fnn | integrated_gradients_steps100 | negative | 초반엔 | 산으로 | 어이없음 | 0.565642 |
| fnn | integrated_gradients_steps50 | negative | 초반엔 | 산으로 | 어이없음 | 0.565461 |
| fnn | lime | negative | 갈수록 | 산으로 | 어이없음 | 0.55224 |
| fnn | occlusion | negative | 초반엔 | 오? | 어이없음 | 0.541674 |
| cnn | integrated_gradients_steps50 | negative | 오? | 산으로 | 어이없음 | 0.533411 |
| cnn | integrated_gradients_steps100 | negative | 오? | 산으로 | 어이없음 | 0.533398 |
| cnn | unigram_occlusion | negative | 오? | 산으로 | 어이없음 | 0.518094 |
| cnn | ngram_occlusion | negative | 산으로 | ㅋㅋㅋ | 어이없음 | 0.456453 |
| transformer | occlusion | negative | 산으로 | ㅋㅋㅋ | 어이없음 | 0.417724 |

## case_022

text: 인생작입니다. 살면서 본 시트콤 중에 감히 최고라고 말하고 싶네요.

- LLM sentiment: positive
- LLM evidence: 인생작입니다., 감히 최고라고 말하고 싶네요.
- LLM reason: 인생작, 최고라는 표현으로 작품을 매우 긍정적으로 평가합니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | occlusion | positive | 인생작입니다. | 시트콤 | 최고라고 | 0.518489 |
| cnn | unigram_occlusion | positive | 인생작입니다. | 살면서 | 최고라고 | 0.510137 |
| cnn | filter_activation | positive | 인생작입니다. | 시트콤 | 최고라고 | 0.499344 |
| cnn | ngram_occlusion | positive | 중에 | 감히 | 최고라고 | 0.496862 |
| transformer | integrated_gradients_steps100 | positive | 인생작입니다. | 시트콤 | 최고라고 | 0.480411 |
| cnn | integrated_gradients_steps100 | positive | 인생작입니다. | 시트콤 | 최고라고 | 0.477618 |
| cnn | integrated_gradients_steps50 | positive | 인생작입니다. | 시트콤 | 최고라고 | 0.47739 |
| transformer | integrated_gradients_steps50 | positive | 인생작입니다. | 시트콤 | 최고라고 | 0.460597 |
| fnn | lime | positive | 인생작입니다. | 중에 | 최고라고 | 0.441438 |
| fnn | integrated_gradients_steps50 | positive | 인생작입니다. | 중에 | 최고라고 | 0.43913 |
| fnn | integrated_gradients_steps100 | positive | 인생작입니다. | 중에 | 최고라고 | 0.437388 |
| fnn | occlusion | positive | 인생작입니다. | 중에 | 최고라고 | 0.434515 |

## case_023

text: 지루해서 중간에 나왔어요 돈날림주의;;;

- LLM sentiment: negative
- LLM evidence: 지루해서, 중간에 나왔어요, 돈날림주의;;;
- LLM reason: 지루해서 중간에 나왔고 돈 낭비라고 표현해 전반적으로 강한 부정 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| fnn | lime | negative | 지루해서 | 중간에 | 돈날림주의;;; | 0.827598 |
| cnn | ngram_occlusion | negative | 중간에 | 나왔어요 | 돈날림주의;;; | 0.823806 |
| cnn | filter_activation | negative | 중간에 | 나왔어요 | 돈날림주의;;; | 0.812839 |
| fnn | integrated_gradients_steps50 | negative | 지루해서 | 나왔어요 | 돈날림주의;;; | 0.79811 |
| fnn | integrated_gradients_steps100 | negative | 지루해서 | 나왔어요 | 돈날림주의;;; | 0.796752 |
| transformer | integrated_gradients_steps100 | negative | 지루해서 | 나왔어요 | 돈날림주의;;; | 0.78187 |
| cnn | unigram_occlusion | negative | 지루해서 | 중간에 | 돈날림주의;;; | 0.77801 |
| fnn | occlusion | negative | 지루해서 | 나왔어요 | 돈날림주의;;; | 0.772698 |
| transformer | integrated_gradients_steps50 | negative | 지루해서 | 나왔어요 | 돈날림주의;;; | 0.770196 |
| cnn | integrated_gradients_steps100 | negative | 지루해서 | 중간에 | 돈날림주의;;; | 0.769094 |
| cnn | integrated_gradients_steps50 | negative | 지루해서 | 중간에 | 돈날림주의;;; | 0.768912 |
| transformer | occlusion | negative | 지루해서 | 나왔어요 | 돈날림주의;;; | 0.653261 |

## case_024

text: 영상미는 좋은데 스토리가 너무 엉성해요. 남는 게 없는 느낌.

- LLM sentiment: negative
- LLM evidence: 영상미는 좋은데, 스토리가 너무 엉성해요., 남는 게 없는 느낌.
- LLM reason: 영상미는 좋다고 했지만 스토리가 엉성하고 남는 게 없다고 평가해 전체적으로 부정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps50 | negative | 너무 | 엉성해요. | 없는 | 0.572637 |
| transformer | integrated_gradients_steps100 | negative | 너무 | 엉성해요. | 없는 | 0.570829 |
| cnn | unigram_occlusion | negative | 스토리가 | 엉성해요. | 없는 | 0.552743 |
| cnn | ngram_occlusion | negative | 스토리가 | 너무 | 엉성해요. | 0.543572 |
| cnn | integrated_gradients_steps50 | negative | 좋은데 | 너무 | 엉성해요. | 0.543115 |
| cnn | integrated_gradients_steps100 | negative | 좋은데 | 너무 | 엉성해요. | 0.542952 |
| cnn | filter_activation | negative | 엉성해요. | 없는 | 느낌. | 0.491633 |
| transformer | occlusion | negative | 영상미는 | 좋은데 | 스토리가 | 0.370527 |

## case_025

text: 킬링타임으로 제격임 ㅋㅋ 유치하긴 한데 가볍게 웃기 좋네요

- LLM sentiment: positive
- LLM evidence: 킬링타임으로 제격임, 유치하긴 한데, 가볍게 웃기 좋네요
- LLM reason: 유치하다는 아쉬움은 있지만, 킬링타임으로 적합하고 가볍게 웃기 좋다는 평가가 중심이라 긍정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| fnn | lime | positive | 제격임 | 유치하긴 | 좋네요 | 0.580597 |
| fnn | occlusion | positive | 유치하긴 | 한데 | 좋네요 | 0.570409 |
| fnn | integrated_gradients_steps50 | positive | 제격임 | 유치하긴 | 좋네요 | 0.567605 |
| fnn | integrated_gradients_steps100 | positive | 제격임 | 유치하긴 | 좋네요 | 0.567228 |
| transformer | integrated_gradients_steps50 | positive | 제격임 | 유치하긴 | 좋네요 | 0.536708 |
| transformer | integrated_gradients_steps100 | positive | 제격임 | 유치하긴 | 좋네요 | 0.528932 |
| cnn | unigram_occlusion | positive | 유치하긴 | 가볍게 | 좋네요 | 0.527494 |
| transformer | occlusion | positive | 제격임 | 한데 | 좋네요 | 0.526787 |
| cnn | integrated_gradients_steps100 | positive | 제격임 | 유치하긴 | 좋네요 | 0.511088 |
| cnn | integrated_gradients_steps50 | positive | 제격임 | 유치하긴 | 좋네요 | 0.510978 |
| cnn | filter_activation | positive | 유치하긴 | 한데 | 가볍게 | 0.402652 |
| cnn | ngram_occlusion | positive | 제격임 | ㅋㅋ | 유치하긴 | 0.377507 |

## case_026

text: 평점 왜 이럼? 생각보다 완전 명작인데 억까들 많네

- LLM sentiment: positive
- LLM evidence: 생각보다 완전 명작인데, 억까들 많네, 평점 왜 이럼?
- LLM reason: 영화를 '완전 명작'이라고 칭하고 낮은 평점을 부당한 비난 탓으로 돌려 전반적으로 긍정적이다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps50 | positive | 생각보다 | 완전 | 명작인데 | 0.565251 |
| transformer | integrated_gradients_steps100 | positive | 생각보다 | 완전 | 명작인데 | 0.561772 |
| cnn | ngram_occlusion | positive | 생각보다 | 완전 | 명작인데 | 0.552578 |
| fnn | occlusion | positive | 왜 | 명작인데 | 억까들 | 0.52507 |
| fnn | integrated_gradients_steps100 | positive | 명작인데 | 억까들 | 많네 | 0.522233 |
| fnn | integrated_gradients_steps50 | positive | 명작인데 | 억까들 | 많네 | 0.521045 |
| fnn | lime | positive | 왜 | 명작인데 | 억까들 | 0.512394 |
| cnn | unigram_occlusion | positive | 왜 | 완전 | 명작인데 | 0.506176 |
| cnn | integrated_gradients_steps50 | positive | 왜 | 명작인데 | 억까들 | 0.505567 |
| cnn | integrated_gradients_steps100 | positive | 왜 | 명작인데 | 억까들 | 0.505301 |
| transformer | occlusion | positive | 이럼? | 명작인데 | 억까들 | 0.493684 |
| cnn | filter_activation | positive | 이럼? | 완전 | 명작인데 | 0.471666 |

## case_027

text: 원작 웹툰이 훨씬 나음.. 실사화는 그냥 폭망 수준

- LLM sentiment: negative
- LLM evidence: 원작 웹툰이 훨씬 나음.., 실사화는 그냥 폭망 수준
- LLM reason: 원작보다 못하며 실사화가 폭망 수준이라고 평가한 부정적 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | filter_activation | negative | 그냥 | 폭망 | 수준 | 0.611696 |
| cnn | ngram_occlusion | negative | 그냥 | 폭망 | 수준 | 0.588952 |
| transformer | integrated_gradients_steps50 | negative | 실사화는 | 폭망 | 수준 | 0.562594 |
| transformer | integrated_gradients_steps100 | negative | 실사화는 | 폭망 | 수준 | 0.562226 |
| fnn | integrated_gradients_steps50 | negative | 그냥 | 폭망 | 수준 | 0.550061 |
| fnn | integrated_gradients_steps100 | negative | 나음.. | 그냥 | 폭망 | 0.549002 |
| cnn | unigram_occlusion | negative | 웹툰이 | 그냥 | 수준 | 0.542899 |
| transformer | occlusion | negative | 실사화는 | 그냥 | 폭망 | 0.53852 |
| fnn | lime | negative | 훨씬 | 폭망 | 수준 | 0.538509 |
| cnn | integrated_gradients_steps50 | negative | 웹툰이 | 그냥 | 수준 | 0.527452 |
| cnn | integrated_gradients_steps100 | negative | 웹툰이 | 그냥 | 수준 | 0.52715 |
| fnn | occlusion | negative | 웹툰이 | 그냥 | 폭망 | 0.519189 |

## case_028

text: 가볍게 볼 로맨틱 코미디 찾다가 완전 인생 영화 건짐... 설렘 터짐

- LLM sentiment: positive
- LLM evidence: 완전 인생 영화 건짐..., 설렘 터짐
- LLM reason: 영화를 '인생 영화'라고 평가하고 설렘이 크다고 표현해 전반적으로 매우 긍정적인 리뷰입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | occlusion | positive | 인생 | 영화 | 설렘 | 0.636243 |
| cnn | integrated_gradients_steps100 | positive | 인생 | 건짐... | 설렘 | 0.549677 |
| cnn | integrated_gradients_steps50 | positive | 인생 | 건짐... | 설렘 | 0.548984 |
| transformer | integrated_gradients_steps100 | positive | 찾다가 | 인생 | 설렘 | 0.501606 |
| transformer | integrated_gradients_steps50 | positive | 찾다가 | 인생 | 설렘 | 0.494649 |
| fnn | occlusion | positive | 볼 | 인생 | 설렘 | 0.444812 |
| cnn | filter_activation | positive | 로맨틱 | 인생 | 영화 | 0.425894 |
| cnn | ngram_occlusion | positive | 찾다가 | 완전 | 인생 | 0.42337 |
| cnn | unigram_occlusion | positive | 볼 | 인생 | 설렘 | 0.418611 |
| fnn | integrated_gradients_steps100 | positive | 볼 | 인생 | 설렘 | 0.409004 |
| fnn | integrated_gradients_steps50 | positive | 볼 | 인생 | 설렘 | 0.408154 |
| fnn | lime | positive | 볼 | 건짐... | 설렘 | 0.341977 |

## case_029

text: 1점도 아깝다 스토리 연출 캐스팅 다 엉망진창임

- LLM sentiment: negative
- LLM evidence: 1점도 아깝다, 스토리 연출 캐스팅 다 엉망진창임
- LLM reason: 1점도 아깝고 모든 요소가 엉망이라는 강한 부정 평가이다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| cnn | ngram_occlusion | negative | 1점도 | 아깝다 | 엉망진창임 | 0.640753 |
| cnn | filter_activation | negative | 1점도 | 아깝다 | 스토리 | 0.616066 |
| fnn | lime | negative | 1점도 | 아깝다 | 엉망진창임 | 0.608818 |
| transformer | integrated_gradients_steps50 | negative | 스토리 | 연출 | 엉망진창임 | 0.606901 |
| transformer | integrated_gradients_steps100 | negative | 스토리 | 연출 | 엉망진창임 | 0.604914 |
| fnn | integrated_gradients_steps100 | negative | 1점도 | 아깝다 | 엉망진창임 | 0.584926 |
| fnn | integrated_gradients_steps50 | negative | 1점도 | 아깝다 | 엉망진창임 | 0.583626 |
| cnn | unigram_occlusion | negative | 1점도 | 아깝다 | 엉망진창임 | 0.575085 |
| cnn | integrated_gradients_steps100 | negative | 1점도 | 아깝다 | 엉망진창임 | 0.571033 |
| cnn | integrated_gradients_steps50 | negative | 1점도 | 아깝다 | 엉망진창임 | 0.570967 |
| fnn | occlusion | negative | 1점도 | 아깝다 | 엉망진창임 | 0.544305 |
| transformer | occlusion | negative | 아깝다 | 캐스팅 | 다 | 0.440046 |

## case_030

text: 가슴 아픈 역사적 사실을 잘 담아낸 의미 있는 영화입니다. 먹먹하네요.

- LLM sentiment: positive
- LLM evidence: 잘 담아낸, 의미 있는 영화입니다., 먹먹하네요.
- LLM reason: 아픈 역사를 다룬 무거운 내용이지만, 잘 담아낸 의미 있는 영화라고 평가해 전반적으로 긍정적입니다.

| model | method | prediction | top words | score |
| --- | --- | --- | --- | ---: |
| transformer | integrated_gradients_steps100 | positive | 담아낸 | 있는 | 영화입니다. | 0.624957 |
| transformer | integrated_gradients_steps50 | positive | 담아낸 | 있는 | 영화입니다. | 0.6176 |
| transformer | occlusion | positive | 담아낸 | 영화입니다. | 먹먹하네요. | 0.550324 |
| fnn | integrated_gradients_steps100 | positive | 가슴 | 사실을 | 영화입니다. | 0.291558 |
| fnn | integrated_gradients_steps50 | positive | 가슴 | 사실을 | 영화입니다. | 0.289911 |
| fnn | occlusion | positive | 가슴 | 사실을 | 영화입니다. | 0.284695 |
| cnn | ngram_occlusion | positive | 역사적 | 사실을 | 영화입니다. | 0.279517 |
| cnn | unigram_occlusion | positive | 가슴 | 사실을 | 영화입니다. | 0.272939 |
| cnn | filter_activation | positive | 아픈 | 역사적 | 사실을 | 0.134708 |
| cnn | integrated_gradients_steps100 | positive | 가슴 | 아픈 | 사실을 | 0.088975 |
| cnn | integrated_gradients_steps50 | positive | 가슴 | 아픈 | 사실을 | 0.088292 |
| fnn | lime | positive | 가슴 | 아픈 | 사실을 | 0.057872 |
