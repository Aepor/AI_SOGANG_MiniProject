# CNN XAI Case Summary

이 파일은 CNN XAI 모듈형 파이프라인이 생성한 사례 요약이다.

## test_29736 (TP)

- text: 굳굳굳굳굳
- true label: 긍정
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0000
- positive prob: 1.0000
- tokens: 굳다 / 굳다 / 굳다 / 굳다 / 굳다

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 굳다 | 0 | 0.0001 |
| 굳다 | 3 | 0.0001 |
| 굳다 | 2 | 0.0001 |
| 굳다 | 4 | 0.0001 |
| 굳다 | 1 | 0.0001 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 5 | 굳다 굳다 굳다 굳다 굳다 | 0 | 0.5453 |
| 4 | 굳다 굳다 굳다 굳다 | 1 | 0.0513 |
| 4 | 굳다 굳다 굳다 굳다 | 0 | 0.0272 |
| 3 | 굳다 굳다 굳다 | 1 | 0.0102 |
| 3 | 굳다 굳다 굳다 | 2 | 0.0030 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 4 | 37 | 굳다 굳다 굳다 굳다 | 0.4473 |
| 4 | 10 | 굳다 굳다 굳다 굳다 | 0.4101 |
| 5 | 72 | 굳다 굳다 굳다 굳다 굳다 | 0.3809 |
| 5 | 71 | 굳다 굳다 굳다 굳다 굳다 | 0.3419 |
| 5 | 96 | 굳다 굳다 굳다 굳다 굳다 | 0.2809 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 굳다 | 1 | 1.9008 | 1.0000 |
| 굳다 | 0 | 1.8638 | 0.9805 |
| 굳다 | 3 | 1.5899 | 0.8364 |
| 굳다 | 2 | 0.9244 | 0.4863 |
| 굳다 | 4 | 0.4762 | 0.2505 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 굳다 | 1 | 1.7162 | 1.0000 |
| 굳다 | 0 | 1.5849 | 0.9235 |
| 굳다 | 3 | 1.4047 | 0.8185 |
| 굳다 | 2 | 0.8901 | 0.5186 |
| 굳다 | 4 | 0.5715 | 0.3330 |

## test_14941 (TP)

- text: 짱이다!짱짱!너무재밌고멋지다!
- true label: 긍정
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0000
- positive prob: 1.0000
- tokens: 짱 / 이다 / ! / 짱짱 / ! / 너무 / 재밌다 / 멋지다 / !

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 재밌다 | 6 | 0.0011 |
| 짱짱 | 3 | 0.0002 |
| 멋지다 | 7 | 0.0001 |
| ! | 8 | 0.0000 |
| 짱 | 0 | 0.0000 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 5 | 짱짱 ! 너무 재밌다 멋지다 | 3 | 0.0195 |
| 5 | ! 짱짱 ! 너무 재밌다 | 2 | 0.0096 |
| 4 | 짱짱 ! 너무 재밌다 | 3 | 0.0071 |
| 2 | 재밌다 멋지다 | 6 | 0.0025 |
| 3 | 재밌다 멋지다 ! | 6 | 0.0021 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 5 | 72 | 짱짱 ! 너무 재밌다 멋지다 | 0.6569 |
| 5 | 69 | 짱짱 ! 너무 재밌다 멋지다 | 0.2035 |
| 3 | 82 | 짱짱 ! 너무 | 0.1946 |
| 4 | 37 | 너무 재밌다 멋지다 ! | 0.1926 |
| 3 | 28 | 짱 이다 ! | 0.1917 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 재밌다 | 6 | 1.7490 | 1.0000 |
| 짱짱 | 3 | 0.9018 | 0.5156 |
| 멋지다 | 7 | 0.7307 | 0.4178 |
| ! | 8 | 0.5047 | 0.2886 |
| 짱 | 0 | 0.3668 | 0.2097 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 재밌다 | 6 | 1.6243 | 1.0000 |
| 짱짱 | 3 | 0.8520 | 0.5245 |
| 멋지다 | 7 | 0.7226 | 0.4449 |
| ! | 8 | 0.5158 | 0.3176 |
| 짱 | 0 | 0.3072 | 0.1891 |

## test_19341 (TP)

- text: 재밌게 본 공포영화중 하나~굿굿! 정말 보고나서 여운이 남는영화였음ㅋ
- true label: 긍정
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0001
- positive prob: 0.9999
- tokens: 재밌다 / 보다 / 공포영화 / 중 / 하나 / ~ / 굿굿 / ! / 정말 / 보다 / 여운 / 남다 / 영화 / 이다 / ㅋ

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 굿굿 | 6 | 0.0013 |
| 여운 | 10 | 0.0005 |
| 영화 | 12 | 0.0001 |
| 재밌다 | 0 | 0.0001 |
| 남다 | 11 | 0.0001 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 5 | 굿굿 ! 정말 보다 여운 | 6 | 0.0361 |
| 4 | ~ 굿굿 ! 정말 | 5 | 0.0015 |
| 3 | ~ 굿굿 ! | 5 | 0.0014 |
| 5 | ~ 굿굿 ! 정말 보다 | 5 | 0.0014 |
| 3 | 굿굿 ! 정말 | 6 | 0.0013 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 48 | 굿굿 ! 정말 | 0.2994 |
| 4 | 10 | 여운 남다 영화 이다 | 0.2772 |
| 5 | 96 | 여운 남다 영화 이다 ㅋ | 0.2768 |
| 5 | 69 | 굿굿 ! 정말 보다 여운 | 0.2387 |
| 3 | 28 | 굿굿 ! 정말 | 0.2086 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 굿굿 | 6 | 2.4841 | 1.0000 |
| 여운 | 10 | 1.4871 | 0.5987 |
| ! | 7 | 0.7972 | 0.3209 |
| 남다 | 11 | 0.6462 | 0.2601 |
| 재밌다 | 0 | 0.4264 | 0.1716 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 굿굿 | 6 | 2.2065 | 1.0000 |
| 여운 | 10 | 1.3324 | 0.6039 |
| ! | 7 | 0.7249 | 0.3285 |
| 남다 | 11 | 0.6215 | 0.2817 |
| 재밌다 | 0 | 0.3790 | 0.1718 |

## test_16046 (TN)

- text: 최악이다 최악최악최악
- true label: 부정
- predicted label: 부정
- target class: 부정
- negative prob: 1.0000
- positive prob: 0.0000
- tokens: 최악 / 이다 / 최악 / 최악 / 최악

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 최악 | 0 | 0.0000 |
| 최악 | 3 | 0.0000 |
| 최악 | 4 | 0.0000 |
| 최악 | 2 | 0.0000 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 5 | 최악 이다 최악 최악 최악 | 0 | 0.4547 |
| 3 | 최악 최악 최악 | 2 | 0.0005 |
| 4 | 이다 최악 최악 최악 | 1 | 0.0002 |
| 4 | 최악 이다 최악 최악 | 0 | 0.0001 |
| 3 | 이다 최악 최악 | 1 | 0.0000 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 5 | 38 | 최악 이다 최악 최악 최악 | 0.3701 |
| 5 | 90 | 최악 이다 최악 최악 최악 | 0.3680 |
| 3 | 6 | 최악 최악 최악 | 0.3561 |
| 5 | 59 | 최악 이다 최악 최악 최악 | 0.3108 |
| 3 | 10 | 최악 이다 최악 | 0.3088 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 최악 | 4 | 2.3319 | 1.0000 |
| 최악 | 3 | 2.0045 | 0.8596 |
| 최악 | 2 | 1.8314 | 0.7854 |
| 최악 | 0 | 1.6392 | 0.7029 |
| 이다 | 1 | 0.2134 | 0.0915 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 최악 | 4 | 2.1852 | 1.0000 |
| 최악 | 3 | 1.8379 | 0.8411 |
| 최악 | 2 | 1.7264 | 0.7900 |
| 최악 | 0 | 1.4837 | 0.6790 |
| 이다 | 1 | 0.2215 | 0.1014 |

## test_14112 (TN)

- text: 별반개도아깝다OO
- true label: 부정
- predicted label: 부정
- target class: 부정
- negative prob: 0.9999
- positive prob: 0.0001
- tokens: 별 / 반개 / 아깝다 / OO

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 반개 | 1 | 0.0072 |
| 아깝다 | 2 | 0.0011 |
| OO | 3 | 0.0001 |
| 별 | 0 | 0.0000 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 4 | 별 반개 아깝다 OO | 0 | 0.4546 |
| 3 | 반개 아깝다 OO | 1 | 0.3637 |
| 3 | 별 반개 아깝다 | 0 | 0.1363 |
| 2 | 반개 아깝다 | 1 | 0.1059 |
| 2 | 별 반개 | 0 | 0.0098 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 10 | 반개 아깝다 OO | 0.3641 |
| 4 | 59 | 별 반개 아깝다 OO | 0.2259 |
| 3 | 80 | 반개 아깝다 OO | 0.1972 |
| 3 | 86 | 별 반개 아깝다 | 0.1896 |
| 3 | 6 | 별 반개 아깝다 | 0.1753 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 반개 | 1 | 2.5257 | 1.0000 |
| 아깝다 | 2 | 2.3381 | 0.9257 |
| OO | 3 | 0.4738 | 0.1876 |
| 별 | 0 | -0.0935 | 0.0370 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 반개 | 1 | 2.2251 | 1.0000 |
| 아깝다 | 2 | 2.0915 | 0.9400 |
| OO | 3 | 0.4671 | 0.2099 |
| 별 | 0 | -0.0935 | 0.0420 |

## test_39367 (TN)

- text: 최악 지루
- true label: 부정
- predicted label: 부정
- target class: 부정
- negative prob: 0.9999
- positive prob: 0.0001
- tokens: 최악 / 지루

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 최악 | 0 | 0.0100 |
| 지루 | 1 | 0.0001 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 2 | 최악 지루 | 0 | 0.4546 |
| 1 | 최악 | 0 | 0.0100 |
| 1 | 지루 | 1 | 0.0001 |

### Top max-pooling contributions

없음


### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 최악 | 0 | 3.8640 | 1.0000 |
| 지루 | 1 | 1.3481 | 0.3489 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 최악 | 0 | 3.3738 | 1.0000 |
| 지루 | 1 | 1.1052 | 0.3276 |

## test_45299 (FP)

- text: 완전 꿀잼이네ㅋㅋㅋㅋ
- true label: 부정
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0036
- positive prob: 0.9964
- tokens: 완전 / 꿀잼 / 이네 / ㅋㅋㅋㅋ

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 꿀잼 | 1 | 0.3689 |
| 완전 | 0 | 0.0065 |
| 이네 | 2 | 0.0016 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 2 | 완전 꿀잼 | 0 | 0.5784 |
| 4 | 완전 꿀잼 이네 ㅋㅋㅋㅋ | 0 | 0.5417 |
| 3 | 완전 꿀잼 이네 | 0 | 0.5053 |
| 1 | 꿀잼 | 1 | 0.3689 |
| 2 | 꿀잼 이네 | 1 | 0.3246 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 4 | 37 | 완전 꿀잼 이네 ㅋㅋㅋㅋ | 0.3497 |
| 3 | 11 | 완전 꿀잼 이네 | 0.1681 |
| 3 | 48 | 꿀잼 이네 ㅋㅋㅋㅋ | 0.1563 |
| 4 | 32 | 완전 꿀잼 이네 ㅋㅋㅋㅋ | 0.1479 |
| 4 | 69 | 완전 꿀잼 이네 ㅋㅋㅋㅋ | 0.1237 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 꿀잼 | 1 | 2.6767 | 1.0000 |
| 완전 | 0 | 1.0593 | 0.3957 |
| ㅋㅋㅋㅋ | 3 | -0.3978 | 0.1486 |
| 이네 | 2 | 0.2511 | 0.0938 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 꿀잼 | 1 | 2.1552 | 1.0000 |
| 완전 | 0 | 0.8391 | 0.3893 |
| 이네 | 2 | 0.3428 | 0.1591 |
| ㅋㅋㅋㅋ | 3 | -0.2650 | 0.1230 |

## test_43662 (FP)

- text: 사람들 왈, 동물은 하늘나라에 갈 수 없다고들 생각하지만, 그들이 우리 인간보다 훨씬 먼저 하늘나라에서 살고 있을 수도 있다.
- true label: 부정
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0049
- positive prob: 0.9951
- tokens: 사람 / 왈 / , / 동물 / 하늘나라 / 갈다 / 수 / 없다 / 들다 / 생각 / , / 그 / 우리 / 인간 / 보다 / 훨씬 / 먼저 / 하늘나라 / 에서 / 살 / 고 / 있다 / 수도 / 있다 / .

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 인간 | 13 | 0.0107 |
| 수 | 6 | 0.0071 |
| 먼저 | 16 | 0.0069 |
| 보다 | 14 | 0.0045 |
| 우리 | 12 | 0.0039 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 5 | 우리 인간 보다 훨씬 먼저 | 12 | 0.0720 |
| 5 | 인간 보다 훨씬 먼저 하늘나라 | 13 | 0.0492 |
| 5 | , 그 우리 인간 보다 | 10 | 0.0461 |
| 5 | 생각 , 그 우리 인간 | 9 | 0.0443 |
| 4 | 그 우리 인간 보다 | 11 | 0.0439 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 5 | 7 | 하늘나라 갈다 수 없다 들다 | 0.2230 |
| 5 | 53 | 인간 보다 훨씬 먼저 하늘나라 | 0.1830 |
| 4 | 37 | 우리 인간 보다 훨씬 | 0.1748 |
| 5 | 88 | 하늘나라 갈다 수 없다 들다 | 0.1497 |
| 5 | 96 | 인간 보다 훨씬 먼저 하늘나라 | 0.1440 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 인간 | 13 | 0.7877 | 1.0000 |
| 먼저 | 16 | 0.6829 | 0.8670 |
| 우리 | 12 | 0.5919 | 0.7515 |
| 없다 | 7 | -0.4514 | 0.5731 |
| 보다 | 14 | 0.3517 | 0.4464 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 인간 | 13 | 0.6171 | 1.0000 |
| 먼저 | 16 | 0.5661 | 0.9173 |
| 우리 | 12 | 0.4755 | 0.7705 |
| 보다 | 14 | 0.3166 | 0.5131 |
| 없다 | 7 | -0.3017 | 0.4888 |

## test_44058 (FP)

- text: 90년대중반 게임을 즐겨본 나는 영화로 나왔을때 조금...... 재미있었다.
- true label: 부정
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0267
- positive prob: 0.9733
- tokens: 90년 / 대중반 / 게임 / 을 / 즐기다 / 보다 / 나 / 영화로 / 나오다 / 때 / 조금 / ...... / 재미있다 / .

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 재미있다 | 12 | 0.2254 |
| 조금 | 10 | 0.0197 |
| 보다 | 5 | 0.0188 |
| 때 | 9 | 0.0159 |
| 즐기다 | 4 | 0.0134 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 4 | 때 조금 ...... 재미있다 | 9 | 0.4428 |
| 5 | 때 조금 ...... 재미있다 . | 9 | 0.4224 |
| 3 | 조금 ...... 재미있다 | 10 | 0.3457 |
| 5 | 나오다 때 조금 ...... 재미있다 | 8 | 0.3422 |
| 4 | 조금 ...... 재미있다 . | 10 | 0.3172 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 68 | 게임 을 즐기다 | 0.2189 |
| 5 | 72 | 90년 대중반 게임 을 즐기다 | 0.1770 |
| 4 | 10 | 조금 ...... 재미있다 . | 0.1359 |
| 4 | 69 | 대중반 게임 을 즐기다 | 0.1347 |
| 3 | 45 | 대중반 게임 을 | 0.1298 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 재미있다 | 12 | 0.9206 | 1.0000 |
| 조금 | 10 | 0.7496 | 0.8143 |
| 게임 | 2 | -0.4755 | 0.5166 |
| 때 | 9 | 0.3854 | 0.4187 |
| 나오다 | 8 | -0.3633 | 0.3947 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 재미있다 | 12 | 0.9530 | 1.0000 |
| 조금 | 10 | 0.5483 | 0.5754 |
| 게임 | 2 | -0.3481 | 0.3653 |
| 을 | 3 | 0.3182 | 0.3339 |
| 때 | 9 | 0.2971 | 0.3118 |

## test_6533 (FN)

- text: 이거 송어 찍은 감독 맞어? 정말 실망이다.
- true label: 긍정
- predicted label: 부정
- target class: 부정
- negative prob: 0.9979
- positive prob: 0.0021
- tokens: 거 / 송어 / 찍다 / 감독 / 맞다 / ? / 정말 / 실망 / 이다 / .

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 실망 | 7 | 0.0649 |
| ? | 5 | 0.0030 |
| 찍다 | 2 | 0.0022 |
| 감독 | 3 | 0.0008 |
| 맞다 | 4 | 0.0008 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 5 | 감독 맞다 ? 정말 실망 | 3 | 0.2514 |
| 4 | 맞다 ? 정말 실망 | 4 | 0.1262 |
| 1 | 실망 | 7 | 0.0649 |
| 3 | ? 정말 실망 | 5 | 0.0584 |
| 5 | 맞다 ? 정말 실망 이다 | 4 | 0.0576 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 86 | 정말 실망 이다 | 0.3536 |
| 3 | 56 | 송어 찍다 감독 | 0.1669 |
| 5 | 95 | ? 정말 실망 이다 . | 0.1654 |
| 3 | 17 | 실망 이다 . | 0.1648 |
| 3 | 91 | 거 송어 찍다 | 0.1618 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 실망 | 7 | 2.9511 | 1.0000 |
| 찍다 | 2 | 0.8029 | 0.2721 |
| ? | 5 | 0.3629 | 0.1230 |
| 송어 | 1 | -0.2481 | 0.0841 |
| . | 9 | 0.1910 | 0.0647 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 실망 | 7 | 2.4594 | 1.0000 |
| 찍다 | 2 | 0.6515 | 0.2649 |
| ? | 5 | 0.2969 | 0.1207 |
| 송어 | 1 | -0.1686 | 0.0686 |
| . | 9 | 0.1473 | 0.0599 |

## test_28966 (FN)

- text: ㅋㅋㅋ 스파이크 강시브리시브테니스 테니스
- true label: 긍정
- predicted label: 부정
- target class: 부정
- negative prob: 0.9900
- positive prob: 0.0100
- tokens: ㅋㅋㅋ / 스파이크 / 강시 / 브리 / 시브 / 테니스 / 테니스

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 스파이크 | 1 | 0.6660 |
| 강시 | 2 | 0.0081 |
| 브리 | 3 | 0.0008 |
| ㅋㅋㅋ | 0 | 0.0001 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 3 | ㅋㅋㅋ 스파이크 강시 | 0 | 0.7349 |
| 2 | 스파이크 강시 | 1 | 0.7272 |
| 2 | ㅋㅋㅋ 스파이크 | 0 | 0.6761 |
| 1 | 스파이크 | 1 | 0.6660 |
| 4 | 스파이크 강시 브리 시브 | 1 | 0.6628 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 10 | 스파이크 강시 브리 | 0.3518 |
| 3 | 80 | 스파이크 강시 브리 | 0.2840 |
| 3 | 30 | 스파이크 강시 브리 | 0.2750 |
| 3 | 73 | 스파이크 강시 브리 | 0.2138 |
| 3 | 6 | ㅋㅋㅋ 스파이크 강시 | 0.2022 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 스파이크 | 1 | 3.6125 | 1.0000 |
| 브리 | 3 | -0.7575 | 0.2097 |
| 테니스 | 5 | -0.3004 | 0.0831 |
| 시브 | 4 | -0.2014 | 0.0557 |
| 테니스 | 6 | -0.1938 | 0.0536 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 스파이크 | 1 | 3.0268 | 1.0000 |
| 브리 | 3 | -0.4873 | 0.1610 |
| 테니스 | 5 | -0.2083 | 0.0688 |
| 시브 | 4 | -0.1812 | 0.0599 |
| 강시 | 2 | 0.1636 | 0.0541 |

## test_28405 (FN)

- text: 이재용감독의발연기bbb
- true label: 긍정
- predicted label: 부정
- target class: 부정
- negative prob: 0.9816
- positive prob: 0.0184
- tokens: 이재용 / 감독 / 발연기 / bbb

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 발연기 | 2 | 0.7670 |
| 감독 | 1 | 0.0490 |
| 이재용 | 0 | 0.0010 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 3 | 이재용 감독 발연기 | 0 | 0.9460 |
| 2 | 감독 발연기 | 1 | 0.9398 |
| 1 | 발연기 | 2 | 0.7670 |
| 4 | 이재용 감독 발연기 bbb | 0 | 0.4363 |
| 3 | 감독 발연기 bbb | 1 | 0.4362 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 17 | 감독 발연기 bbb | 0.1727 |
| 3 | 56 | 감독 발연기 bbb | 0.1707 |
| 3 | 86 | 감독 발연기 bbb | 0.1389 |
| 3 | 7 | 감독 발연기 bbb | 0.1294 |
| 3 | 10 | 이재용 감독 발연기 | 0.1100 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 발연기 | 2 | 2.9449 | 1.0000 |
| bbb | 3 | -2.2620 | 0.7681 |
| 감독 | 1 | 0.8348 | 0.2835 |
| 이재용 | 0 | -0.1900 | 0.0645 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 발연기 | 2 | 2.4847 | 1.0000 |
| bbb | 3 | -1.5796 | 0.6358 |
| 감독 | 1 | 0.6966 | 0.2804 |
| 이재용 | 0 | -0.1095 | 0.0441 |

## custom_1 (custom)

- text: 이 영화 진짜 시간 가는 줄 모르고 봤어요
- true label: 
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.2382
- positive prob: 0.7618
- tokens: 영화 / 진짜 / 시간 / 가다 / 줄 / 모르다 / 보다

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 보다 | 6 | 0.1392 |
| 가다 | 3 | 0.0950 |
| 시간 | 2 | 0.0866 |
| 줄 | 4 | 0.0689 |
| 영화 | 0 | 0.0211 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 4 | 가다 줄 모르다 보다 | 3 | 0.4538 |
| 3 | 가다 줄 모르다 | 3 | 0.3180 |
| 2 | 가다 줄 | 3 | 0.2212 |
| 3 | 줄 모르다 보다 | 4 | 0.2015 |
| 5 | 시간 가다 줄 모르다 보다 | 2 | 0.1676 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 1 | 가다 줄 모르다 | 0.1281 |
| 3 | 52 | 진짜 시간 가다 | 0.1277 |
| 3 | 75 | 진짜 시간 가다 | 0.0900 |
| 3 | 42 | 줄 모르다 보다 | 0.0895 |
| 4 | 69 | 진짜 시간 가다 줄 | 0.0795 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 모르다 | 5 | 0.2876 | 1.0000 |
| 가다 | 3 | 0.2776 | 0.9652 |
| 시간 | 2 | 0.2364 | 0.8221 |
| 보다 | 6 | 0.1224 | 0.4258 |
| 영화 | 0 | 0.1148 | 0.3993 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 가다 | 3 | 0.2083 | 1.0000 |
| 시간 | 2 | 0.1777 | 0.8528 |
| 보다 | 6 | 0.1743 | 0.8369 |
| 모르다 | 5 | 0.1558 | 0.7477 |
| 영화 | 0 | 0.0735 | 0.3530 |

## custom_2 (custom)

- text: 완전 최악이고 시간이 아까웠다
- true label: 
- predicted label: 부정
- target class: 부정
- negative prob: 0.9998
- positive prob: 0.0002
- tokens: 완전 / 최악 / 이고 / 시간 / 아깝다

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 최악 | 1 | 0.0352 |
| 아깝다 | 4 | 0.0005 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 4 | 최악 이고 시간 아깝다 | 1 | 0.6822 |
| 5 | 완전 최악 이고 시간 아깝다 | 0 | 0.4546 |
| 3 | 최악 이고 시간 | 1 | 0.0678 |
| 1 | 최악 | 1 | 0.0352 |
| 4 | 완전 최악 이고 시간 | 0 | 0.0292 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 10 | 최악 이고 시간 | 0.3798 |
| 3 | 6 | 완전 최악 이고 | 0.2976 |
| 3 | 66 | 완전 최악 이고 | 0.2386 |
| 3 | 86 | 이고 시간 아깝다 | 0.2124 |
| 4 | 87 | 최악 이고 시간 아깝다 | 0.2081 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 최악 | 1 | 3.9714 | 1.0000 |
| 아깝다 | 4 | 0.5364 | 0.1351 |
| 완전 | 0 | 0.0204 | 0.0051 |
| 시간 | 3 | -0.0066 | 0.0017 |
| 이고 | 2 | -0.0017 | 0.0004 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 최악 | 1 | 3.4450 | 1.0000 |
| 아깝다 | 4 | 0.6417 | 0.1863 |
| 이고 | 2 | 0.1012 | 0.0294 |
| 시간 | 3 | 0.0540 | 0.0157 |
| 완전 | 0 | 0.0420 | 0.0122 |

## custom_3 (custom)

- text: 배우는 좋았지만 스토리는 너무 지루했다
- true label: 
- predicted label: 부정
- target class: 부정
- negative prob: 0.9969
- positive prob: 0.0031
- tokens: 배우다 / 좋다 / 스토리 / 너무 / 지루하다

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 지루하다 | 4 | 0.2642 |
| 너무 | 3 | 0.0153 |
| 스토리 | 2 | 0.0046 |
| 배우다 | 0 | 0.0009 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 3 | 스토리 너무 지루하다 | 2 | 0.7643 |
| 2 | 너무 지루하다 | 3 | 0.5189 |
| 4 | 좋다 스토리 너무 지루하다 | 1 | 0.4618 |
| 5 | 배우다 좋다 스토리 너무 지루하다 | 0 | 0.4516 |
| 1 | 지루하다 | 4 | 0.2642 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 4 | 98 | 배우다 좋다 스토리 너무 | 0.1266 |
| 5 | 90 | 배우다 좋다 스토리 너무 지루하다 | 0.0945 |
| 3 | 64 | 스토리 너무 지루하다 | 0.0935 |
| 3 | 97 | 배우다 좋다 스토리 | 0.0589 |
| 4 | 26 | 배우다 좋다 스토리 너무 | 0.0584 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 지루하다 | 4 | 2.7012 | 1.0000 |
| 너무 | 3 | 0.9333 | 0.3455 |
| 좋다 | 1 | -0.2580 | 0.0955 |
| 배우다 | 0 | 0.1717 | 0.0635 |
| 스토리 | 2 | 0.1273 | 0.0471 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 지루하다 | 4 | 2.2657 | 1.0000 |
| 너무 | 3 | 0.8204 | 0.3621 |
| 배우다 | 0 | 0.1332 | 0.0588 |
| 좋다 | 1 | -0.1255 | 0.0554 |
| 스토리 | 2 | 0.0985 | 0.0435 |

## custom_4 (custom)

- text: 생각보다 지루하지 않고 감동적이었다
- true label: 
- predicted label: 긍정
- target class: 긍정
- negative prob: 0.0274
- positive prob: 0.9726
- tokens: 생각 / 보다 / 지루하다 / 않다 / 감동 / 적 / 이다

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 않다 | 3 | 0.3957 |
| 감동 | 4 | 0.2048 |
| 지루하다 | 2 | 0.0260 |
| 생각 | 0 | 0.0022 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 4 | 않다 감동 적 이다 | 3 | 0.9273 |
| 3 | 않다 감동 적 | 3 | 0.8816 |
| 2 | 않다 감동 | 3 | 0.8632 |
| 5 | 생각 보다 지루하다 않다 감동 | 0 | 0.3985 |
| 1 | 않다 | 3 | 0.3957 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 68 | 보다 지루하다 않다 | 0.2458 |
| 4 | 75 | 보다 지루하다 않다 감동 | 0.0985 |
| 5 | 69 | 생각 보다 지루하다 않다 감동 | 0.0979 |
| 4 | 68 | 생각 보다 지루하다 않다 | 0.0851 |
| 3 | 82 | 감동 적 이다 | 0.0794 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 감동 | 4 | 1.5917 | 1.0000 |
| 않다 | 3 | 0.7730 | 0.4857 |
| 이다 | 6 | 0.3339 | 0.2098 |
| 생각 | 0 | 0.1142 | 0.0717 |
| 보다 | 1 | -0.0580 | 0.0364 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 감동 | 4 | 1.1714 | 1.0000 |
| 않다 | 3 | 0.3892 | 0.3322 |
| 이다 | 6 | 0.2963 | 0.2529 |
| 지루하다 | 2 | 0.1865 | 0.1592 |
| 적 | 5 | 0.1202 | 0.1026 |

## custom_5 (custom)

- text: 재미는 있는데 결말이 별로였다
- true label: 
- predicted label: 부정
- target class: 부정
- negative prob: 0.9601
- positive prob: 0.0399
- tokens: 재미 / 있다 / 결말 / 별로 / 이다

### Top unigram occlusion
| token | position | prob_drop |
| --- | --- | --- |
| 별로 | 3 | 0.7950 |

### Top n-gram occlusion
| n | ngram | start_pos | prob_drop |
| --- | --- | --- | --- |
| 2 | 결말 별로 | 2 | 0.8033 |
| 3 | 결말 별로 이다 | 2 | 0.7979 |
| 1 | 별로 | 3 | 0.7950 |
| 2 | 별로 이다 | 3 | 0.7739 |
| 3 | 있다 결말 별로 | 1 | 0.6904 |

### Top max-pooling contributions
| filter_size | filter_idx | selected_ngram | target_contribution |
| --- | --- | --- | --- |
| 3 | 86 | 결말 별로 이다 | 0.1258 |
| 3 | 80 | 재미 있다 결말 | 0.1046 |
| 4 | 13 | 재미 있다 결말 별로 | 0.0709 |
| 4 | 26 | 재미 있다 결말 별로 | 0.0685 |
| 5 | 38 | 재미 있다 결말 별로 이다 | 0.0654 |

### Top Gradient x Input
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 별로 | 3 | 3.3364 | 1.0000 |
| 재미 | 0 | -0.8038 | 0.2409 |
| 결말 | 2 | -0.1704 | 0.0511 |
| 이다 | 4 | -0.0735 | 0.0220 |
| 있다 | 1 | -0.0698 | 0.0209 |

### Top Integrated Gradients
| token | position | signed_score | normalized_abs_score |
| --- | --- | --- | --- |
| 별로 | 3 | 2.5853 | 1.0000 |
| 재미 | 0 | -0.5922 | 0.2291 |
| 결말 | 2 | -0.1225 | 0.0474 |
| 이다 | 4 | 0.0531 | 0.0205 |
| 있다 | 1 | -0.0465 | 0.0180 |
