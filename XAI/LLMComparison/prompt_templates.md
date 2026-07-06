# LLM Prompt Templates

## llm_sentiment_evidence_v1

The OpenAI API response is used as a qualitative reference explanation, not as
ground truth. The model must judge the review independently from the XAI outputs.

```text
다음 한국어 영화 리뷰가 positive인지 negative인지 판단하세요.

중요:
- 출력은 JSON object 하나만 작성하세요.
- sentiment는 "positive" 또는 "negative" 중 하나만 사용하세요.
- evidence는 최대 3개까지 작성하세요.
- evidence는 반드시 제공된 words 배열의 어절 index로 지정하세요.
- phrase는 원문에 실제로 등장하는 표현을 사용하세요.
- polarity는 "positive" 또는 "negative" 중 하나만 사용하세요.
- strength는 0.0~1.0 사이 숫자로 작성하세요.
- 이 설명은 정답 라벨이 아니라 XAI 비교용 LLM 사후 자연어 기준으로 사용됩니다.

입력:
{
  "sample_id": "...",
  "text": "...",
  "words": ["...", "..."]
}

출력 schema:
{
  "sample_id": "...",
  "text": "...",
  "words": ["...", "..."],
  "sentiment": "positive|negative",
  "evidence": [
    {
      "phrase": "...",
      "word_indices": [0, 1],
      "polarity": "positive|negative",
      "strength": 0.8,
      "reason": "..."
    }
  ],
  "brief_reason": "..."
}
```
