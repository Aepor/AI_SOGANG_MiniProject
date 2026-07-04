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

def explain_integrated_gradients(model, tokenizer, text, max_length=128, device="cpu", steps=100):
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
    
    # WordPiece 토크나이저 감지 시 즉시 에러 발생 후 종료
    try:
        vocab = tokenizer.get_vocab()
        # WordPiece 모델은 보통 수천 개의 '##' 로 시작하는 서브워드를 가집니다.
        # KcELECTRA(BPE)처럼 단순히 '##' 특수기호 1개만 있는 경우는 제외합니다.
        wordpiece_keys = sum(1 for k in vocab.keys() if k.startswith("##"))
        if wordpiece_keys > 10:
            raise ValueError("WordPiece 방식의 토크나이저는 지원하지 않습니다. (BPE 방식만 지원)")
    except ValueError as e:
        raise e
    except Exception:
        pass

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
    
    # 6. 원래 모델의 예측 클래스 확인 및 설명 대상 클래스 고정
    with torch.no_grad():
        original_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(original_outputs.logits, dim=-1).squeeze(0)
        pred_class = int(probs.argmax().item())
        target_class = 1  # 항상 긍정(1) 클래스를 기준으로 중요도를 계산합니다.
    
    # 7. 각 스텝에 대한 기울기(Gradient) 계산 (OOM 방지를 위한 미니배치 처리)
    batch_size = 16
    grads_list = []
    
    for i in range(0, steps, batch_size):
        batch_interpolated_embeds = interpolated_embeds[i : i + batch_size]
        batch_mask = attention_mask.expand(batch_interpolated_embeds.size(0), -1)
        
        batch_outputs = model(inputs_embeds=batch_interpolated_embeds, attention_mask=batch_mask)
        batch_logits = batch_outputs.logits[:, target_class]
        
        batch_grads = torch.autograd.grad(
            outputs=batch_logits,
            inputs=batch_interpolated_embeds,
            grad_outputs=torch.ones_like(batch_logits),
            create_graph=False
        )[0]
        grads_list.append(batch_grads)
        
    grads = torch.cat(grads_list, dim=0)  # Shape: [steps, seq_len, hidden_dim]
    
    # 8. 리만 합(Riemann sum)을 통해 Gradient의 평균 근사치 계산
    avg_grads = grads.mean(dim=0)  # Shape: [seq_len, hidden_dim]
    
    # 9. 최종 Attribution 계산: (input_embeds - baseline_embeds) * avg_grads
    delta = (embeddings - baseline_embeddings).squeeze(0)  # Shape: [seq_len, hidden_dim]
    attributions = avg_grads * delta  # Shape: [seq_len, hidden_dim]
    
    # 10. 최종 토큰 단위 중요도 스칼라값 도출
    token_scores = attributions.sum(dim=-1).detach().cpu().numpy()  # Shape: [seq_len]
    
    # 11. BPE 바이트 디코딩 및 한글 어절 병합
    merged_words = []
    merged_scores = []
    
    # 각 어절의 바이트들을 담을 리스트
    word_bytes_list = []
    
    for i, (token, score) in enumerate(zip(tokens, token_scores)):
        # 특수 토큰 제외
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # 토큰 전처리 (BPE의 Ġ 제거)
        clean_token = token
        if token.startswith("Ġ"):
            clean_token = token[1:]
            
        # 토큰을 바이트 리스트로 변환
        try:
            token_bytes = [BYTE_DECODER[c] for c in clean_token]
        except Exception:
            token_bytes = list(clean_token.encode('utf-8'))
            
        # 서브워드 판별 (BPE 전용: Ġ로 시작하지 않으면 이전 어절의 서브워드로 간주)
        is_subword = not token.startswith("Ġ") and (len(word_bytes_list) > 0)
            
        if is_subword and word_bytes_list:
            word_bytes_list[-1].extend(token_bytes)
            merged_scores[-1] += score
        else:
            word_bytes_list.append(token_bytes)
            merged_scores.append(score)
            
    # 바이트 어절들을 문자열로 디코딩
    for bytes_line in word_bytes_list:
        decoded_word = bytearray(bytes_line).decode('utf-8', errors='ignore').strip()
        merged_words.append(decoded_word)
        
    # 빈 공백 어절 제거 및 해당하는 점수 매칭
    filtered_words = []
    filtered_scores = []
    for w, s in zip(merged_words, merged_scores):
        if w == "":
            continue
        filtered_words.append(w)
        filtered_scores.append(s)
        
    # L1 정규화 (절댓값 총합으로 나누어 비율로 변환)
    total_abs_score = sum(abs(s) for s in filtered_scores)
    if total_abs_score > 0:
        filtered_scores = [s / total_abs_score for s in filtered_scores]
            
    return {
        "words": filtered_words,
        "scores": [float(s) for s in filtered_scores],
        "prediction": pred_class,
        "probability": float(probs[pred_class].item())
    }

if __name__ == "__main__":
    print("explain_ig.py 모듈이 정상 로드되었습니다.")
