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

def explain_occlusion(model, tokenizer, text, max_length=128, device="cpu"):
    """
    텍스트의 각 어절(Word)을 하나씩 가린 뒤, 예측 확률의 변화량(P_orig - P_new)을 기반으로 중요도를 산출합니다.
    """
    # 1. 모델을 eval 모드로 설정
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

    # 2. 입력 텍스트 토큰화 및 디바이스 이동
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True
    )
    input_ids = inputs["input_ids"][0].to(device) # 1D 텐서로 변환
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 3. 원본 입력 문장의 예측 결과 획득
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=inputs["attention_mask"].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        
    pred_class = int(probs.argmax().item())
    target_class = 1  # 항상 긍정(1) 클래스를 기준으로 중요도를 계산합니다.
    p_orig = probs[target_class].item()
    
    # 어절(Word) 단위로 토큰 그룹화
    word_to_token_indices = []
    current_word_indices = []
    
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_word_indices:
                word_to_token_indices.append(current_word_indices)
                current_word_indices = []
            word_to_token_indices.append([i])
        elif token.startswith("Ġ"):
            if current_word_indices:
                word_to_token_indices.append(current_word_indices)
            current_word_indices = [i]
        else:
            if len(current_word_indices) == 0:
                current_word_indices = [i]
            else:
                current_word_indices.append(i)
                
    if current_word_indices:
        word_to_token_indices.append(current_word_indices)
        
    # 4. 각 어절(Word) 단위로 순회하며 Occlusion 계산
    word_scores = []
    merged_words = []
    
    # PAD 토큰 ID
    replace_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.pad_token_id
    
    for word_indices in word_to_token_indices:
        # 이 그룹의 첫 토큰이 특수 토큰인 경우 제외
        first_token = tokens[word_indices[0]]
        if first_token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # 어절 단어 복원 (이 그룹의 토큰들을 병합)
        word_bytes = []
        for idx in word_indices:
            token = tokens[idx]
            clean_token = token
            if token.startswith("Ġ"):
                clean_token = token[1:]
            try:
                token_bytes = [BYTE_DECODER[c] for c in clean_token]
            except Exception:
                token_bytes = list(clean_token.encode('utf-8'))
            word_bytes.extend(token_bytes)
            
        decoded_word = bytearray(word_bytes).decode('utf-8', errors='ignore').strip()
        if decoded_word == "":
            continue
            
        # 해당 어절의 모든 토큰들을 한 번에 Occlude (가림 처리)
        occluded_ids = input_ids.clone()
        # attention_mask는 원본 패딩 정보를 그대로 유지하되, 가린 토큰은 0으로 끄지 않고 1로 유지합니다.
        occluded_mask = inputs["attention_mask"].to(device)
        
        for idx in word_indices:
            occluded_ids[idx] = replace_id
            
        # 모델 예측
        with torch.no_grad():
            occluded_outputs = model(
                input_ids=occluded_ids.unsqueeze(0),
                attention_mask=occluded_mask
            )
            occluded_probs = torch.softmax(occluded_outputs.logits, dim=-1).squeeze(0)
            
        p_new = occluded_probs[target_class].item()
        word_scores.append(p_orig - p_new)
        merged_words.append(decoded_word)
        
    # 5. L1 정규화 (절댓값 총합으로 나누어 비율로 변환)
    total_abs_score = sum(abs(s) for s in word_scores)
    if total_abs_score > 0:
        word_scores = [s / total_abs_score for s in word_scores]
        
    return {
        "words": merged_words,
        "scores": [float(s) for s in word_scores],
        "prediction": pred_class,
        "probability": float(probs[pred_class].item())
    }

if __name__ == "__main__":
    print("explain_occlusion.py 모듈이 정상 로드되었습니다.")
