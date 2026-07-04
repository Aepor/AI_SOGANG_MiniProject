import os
import json
from pathlib import Path
import torch

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.font_manager as fm

    # 윈도우 '맑은 고딕' 폰트의 절대 경로 지정 (두부 현상 해결)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        plt.rc('font', family='Malgun Gothic')
        
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("Warning: matplotlib 또는 seaborn이 설치되지 않아 시각화 이미지는 생성되지 않을 수 있습니다.")

LABELS = {0: "negative", 1: "positive"}

def get_word_level_attention(model, tokenizer, text, device, max_length=128):
    """
    [CLS] 토큰이 각 어절에 미치는 어텐션 가중치를 어절 단위로 병합하고 L1 정규화하여 반환합니다.
    """
    words = text.split()
    
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_id = int(probs.argmax().item())
        probability = float(probs[pred_id].item())
        
    last_layer_attn = outputs.attentions[-1].squeeze(0)  
    avg_attn = last_layer_attn.mean(dim=0)               
    
    # 1. HuggingFace Fast Tokenizer를 이용해 토큰을 어절 단위(word_ids)로 매핑
    try:
        word_ids = encoded.word_ids(batch_index=0)
    except ValueError:
        raise RuntimeError("Fast Tokenizer가 지원되지 않는 모델입니다.")
        
    num_words = len(words)
    word_attn_2d = torch.zeros((num_words, num_words))
    cls_word_attn = torch.zeros(num_words)
    
    word_to_tokens = {w: [] for w in range(num_words)}
    cls_idx = 0
    
    # [CLS] 위치 파악 및 어절별 토큰 인덱스 수집
    for seq_idx, w_id in enumerate(word_ids):
        if w_id is not None and w_id < num_words:
            word_to_tokens[w_id].append(seq_idx)
        elif encoded["input_ids"][0][seq_idx].item() == tokenizer.cls_token_id:
            cls_idx = seq_idx

    # 2. 1D 정규화 (JSON 출력용: [CLS]가 각 어절을 바라보는 어텐션 합산)
    for w in range(num_words):
        if word_to_tokens[w]:
            cls_word_attn[w] = sum(avg_attn[cls_idx, j].item() for j in word_to_tokens[w])

    # 3. 2D 어절 단위 풀링 (히트맵 출력용: 어절 vs 어절 2D 행렬 생성)
    for u in range(num_words):
        tokens_u = word_to_tokens[u]
        if not tokens_u: continue
        for v in range(num_words):
            tokens_v = word_to_tokens[v]
            if not tokens_v: continue
            
            total_attn = sum(avg_attn[i, j].item() for i in tokens_u for j in tokens_v)
            word_attn_2d[u, v] = total_attn / len(tokens_u)

    # 1D JSON L1 정규화
    l1_norm = cls_word_attn.sum().item()
    if l1_norm > 0:
        normalized_scores = (cls_word_attn / l1_norm).tolist()
    else:
        normalized_scores = cls_word_attn.tolist()
        
    # 2D 히트맵 행(Row) 기준 정규화
    row_sums = word_attn_2d.sum(dim=1, keepdim=True)
    word_attn_2d = torch.where(row_sums > 0, word_attn_2d / row_sums, word_attn_2d)

    return {
        "text": text,
        "prediction": LABELS[pred_id],
        "probability": probability,
        "words": words,
        "attention_scores": [round(s, 4) for s in normalized_scores],
        "_word_attn_2d": word_attn_2d.numpy() 
    }

def run_attention_pipeline(model, tokenizer, texts, output_dir, device, max_length, model_name):
    """입력받은 텍스트 목록에 대해 어텐션 분석을 수행하고 히트맵 및 JSON을 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_results = []

    for idx, text in enumerate(texts):
        print(f"\n[어텐션 분석] {text}")
        
        result_dict = get_word_level_attention(model, tokenizer, text, device, max_length)
        
        word_attn_2d = result_dict.pop("_word_attn_2d")
        words = result_dict['words']
        
        json_results.append(result_dict)
        print(f"예측: {result_dict['prediction']} (확률: {result_dict['probability']:.4f})")
        print(f"단어 어텐션: {result_dict['attention_scores']}")

        # 히트맵 그리기
        if 'plt' in globals():
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                word_attn_2d, 
                xticklabels=words, 
                yticklabels=words, 
                cmap="Blues", 
                linewidths=0.5
            )
            plt.title(f"Self-Attention Heatmap (Word Level)\n'{text}'")
            plt.xlabel("Key (어디에 집중했는가)")
            plt.ylabel("Query (어떤 단어가)")
            
            save_path = output_path / f"attention_heatmap_{idx}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"-> 히트맵 이미지 저장: {save_path}")

    # JSON 저장
    model_name_for_file = model_name.split('/')[-1]
    json_path = output_path / f"output_{model_name_for_file}_attention.json"
    json_path.write_text(
        json.dumps(json_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n-> 모든 규격화된 분석 결과가 저장되었습니다: {json_path}")