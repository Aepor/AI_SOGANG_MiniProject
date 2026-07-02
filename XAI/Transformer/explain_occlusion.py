import torch

def merge_subwords_and_scores(tokens, scores):
    """
    특수 토큰([CLS], [SEP], [PAD])을 제외하고, 
    ## 접두어가 붙은 토큰들을 이전 단어에 병합하며 점수를 합산합니다.
    """
    merged_words = []
    merged_scores = []
    
    for token, score in zip(tokens, scores):
        # 1. 특수 토큰 제외
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # 2. ##으로 시작하는 토큰 병합
        if token.startswith("##"):
            if merged_words: # 리스트가 비어있지 않은 경우
                merged_words[-1] += token[2:] # '##' 제거 후 문자열 병합
                merged_scores[-1] += score    # 중요도 점수 합산
            else:
                # 문장 자체가 ## 토큰으로 시작하는 예외적인 경우 처리
                merged_words.append(token[2:])
                merged_scores.append(score)
        # 3. 새로운 단어 추가
        else:
            merged_words.append(token)
            merged_scores.append(score)
            
    return merged_words, merged_scores

def explain_occlusion(model, tokenizer, text, max_length=128, device="cpu"):
    """
    텍스트의 각 토큰을 하나씩 가린 뒤, 예측 확률의 변화량(P_orig - P_new)을 기반으로 중요도를 산출합니다.
    """
    # 1. 모델을 eval 모드로 설정
    model.eval()
    
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
        outputs = model(input_ids.unsqueeze(0))
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        
    pred_class = probs.argmax().item()
    p_orig = probs[pred_class].item()
    
    raw_scores = []
    
    # 보통 단어를 가릴 때는 MASK 토큰을 쓰는 것이 더 자연스러우나, 
    # MASK 토큰이 없는 모델을 대비해 PAD 토큰을 Fallback으로 사용합니다.
    replace_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.pad_token_id
    
    # 4. 각 토큰 위치 i를 순회하며 중요도 계산
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            raw_scores.append(0.0)
            continue
            
        # i번째 토큰의 ID를 교체(Occlusion)
        occluded_ids = input_ids.clone()
        occluded_ids[i] = replace_id
        
        # 변형된 입력으로 모델 예측
        with torch.no_grad():
            occluded_outputs = model(occluded_ids.unsqueeze(0))
            occluded_probs = torch.softmax(occluded_outputs.logits, dim=-1).squeeze(0)
            
        # 원래 예측했던 클래스에 대한 새로운 확률값 추출
        p_new = occluded_probs[pred_class].item()
        
        # 중요도 점수 = 원본 확률 - 가렸을 때의 확률
        raw_scores.append(p_orig - p_new)
        
    # 5. 서브워드 및 점수 병합
    words, scores = merge_subwords_and_scores(tokens, raw_scores)
    
    return {
        "words": words,
        "scores": scores,
        "prediction": pred_class,
        "probability": p_orig
    }

if __name__ == "__main__":
    print("explain_occlusion.py 로드가 완료되었습니다.")
