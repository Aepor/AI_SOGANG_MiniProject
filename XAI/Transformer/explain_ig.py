import torch

def get_byte_decoder():
    """
    GPT-2 / RoBERTa / KcELECTRA BPE 토크나이저에서 사용하는 바이트-유니코드 매핑의 역방향 매핑을 생성합니다.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

# 전역 바이트 디코더 생성
BYTE_DECODER = get_byte_decoder()

def explain_integrated_gradients(model, tokenizer, text, max_length=128, device="cpu", steps=50):
    """
    텍스트에 대해 Integrated Gradients 중요도를 계산하고 한글 어절 단위로 복원 및 병합된 중요도 점수를 반환합니다.
    
    Args:
        model: Hugging Face Sequence Classification 모델
        tokenizer: AutoTokenizer 객체
        text: 설명하려는 단일 문장 (str)
        max_length: 입력 시퀀스 최대 길이
        device: 계산에 사용할 PyTorch 디바이스
        steps: 리만 합(Riemann sum) 적분을 위한 보간 스텝 수 (기본값: 50)
    
    Returns:
        dict: {
            "words": 복원된 단어(어절) 리스트,
            "scores": 각 단어별 Integrated Gradients 기여도 점수,
            "prediction": 모델의 예측 클래스 (0: 부정, 1: 긍정),
            "probability": 예측된 클래스의 확률
        }
    """
    
    # 학습 모드로 전환.
    model.eval()
    
    # 1. 텍스트 토큰화 및 디바이스 이동
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # 2. 토큰 리스트 추출
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # 3. 모델의 임베딩 레이어 확보
    embed_layer = model.get_input_embeddings()
    
    # 4. 입력 임베딩(inputs_embeds) 계산
    with torch.no_grad():
        embeddings = embed_layer(input_ids).clone()  # Shape: [1, seq_len, hidden_dim]
    
    # baseline 정의 (기본적으로 모두 0으로 채워진 Zero Embedding 사용)
    baseline_embeddings = torch.zeros_like(embeddings)
    
    # 5. 경로 상의 보간 임베딩(linear interpolation) 생성
    alphas = torch.linspace(0, 1, steps=steps, device=device)
    # interpolated_embeds shape: [steps, seq_len, hidden_dim]
    interpolated_embeds = baseline_embeddings + alphas[:, None, None] * (embeddings - baseline_embeddings)
    interpolated_embeds.requires_grad_()
    
    # 6. 원래 모델의 예측 클래스 선정
    with torch.no_grad():
        original_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(original_outputs.logits, dim=-1).squeeze(0)
        target_class = int(probs.argmax().item())
    
    # 7. 각 스텝에 대한 기울기(Gradient) 계산
    expanded_mask = attention_mask.expand(steps, -1)
    
    # inputs_embeds를 사용하여 forward 실행
    outputs = model(inputs_embeds=interpolated_embeds, attention_mask=expanded_mask)
    logits = outputs.logits[:, target_class]
    
    # logits 출력값 전체에 대해 interpolated_embeds 의 gradient를 구함
    grads = torch.autograd.grad(
        outputs=logits,
        inputs=interpolated_embeds,
        grad_outputs=torch.ones_like(logits),
        create_graph=False
    )[0]  # Shape: [steps, seq_len, hidden_dim]
    
    # 8. 리만 합(Riemann sum)을 통해 Gradient의 평균 근사치 계산
    avg_grads = grads.mean(dim=0)  # Shape: [seq_len, hidden_dim]
    
    # 9. 최종 Attribution 계산: (input_embeds - baseline_embeds) * avg_grads
    delta = (embeddings - baseline_embeddings).squeeze(0)  # Shape: [seq_len, hidden_dim]
    attributions = avg_grads * delta  # Shape: [seq_len, hidden_dim]
    
    # 10. 최종 토큰 단위 중요도 스칼라값 도출
    token_scores = attributions.sum(dim=-1).detach().cpu().numpy()  # Shape: [seq_len]
    
    # 11. BPE / WordPiece 바이트 디코딩 및 한글 어절 병합
    merged_words = []
    merged_scores = []
    
    for i, (token, score) in enumerate(zip(tokens, token_scores)):
        # 특수 토큰 제외
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # 토큰 문자열을 실제 바이트 배열로 바꾸어 UTF-8 한글 복원
        try:
            byte_array = bytearray([BYTE_DECODER[c] for c in token])
            word_str = byte_array.decode('utf-8', errors='ignore')
        except Exception:
            word_str = token
            
        # 서브워드 판별 및 공백 제거
        if token.startswith("##"):
            # WordPiece subword
            is_subword = True
            clean_word = word_str[2:] if word_str.startswith("##") else word_str
        else:
            # BPE subword: 원래 토큰 형태가 'Ġ'로 시작하지 않는 경우 이전 단어에 붙는 어미/조사로 취급
            # 단, 이전에 생성된 단어가 있어야만 합칠 수 있으므로 len(merged_words) > 0 조건을 사용합니다.
            is_subword = not token.startswith("Ġ") and (len(merged_words) > 0)
            clean_word = word_str.strip()
            
        if is_subword:
            if merged_words:
                merged_words[-1] += clean_word
                merged_scores[-1] += score
            else:
                merged_words.append(clean_word)
                merged_scores.append(score)
        else:
            # 빈 공백 어절 방지
            if clean_word == "":
                continue
            merged_words.append(clean_word)
            merged_scores.append(score)
            
    return {
        "words": merged_words,
        "scores": [float(s) for s in merged_scores],
        "prediction": target_class,
        "probability": float(probs[target_class].item())
    }

if __name__ == "__main__":
    print("explain_ig.py 모듈이 정상 로드되었습니다.")
