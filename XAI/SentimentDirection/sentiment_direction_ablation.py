import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import wilcoxon

# 프로젝트 루트를 PATH에 추가하여 기존 모듈을 가져옵니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from XAI.Transformer.explain_ig import explain_integrated_gradients
    from XAI.Transformer.explain_occlusion import explain_occlusion
except ImportError:
    print("Warning: 기존 XAI 모듈을 불러올 수 없습니다. 경로를 확인해 주세요.")

def load_directions(direction_dir, num_layers):
    directions = {}
    for layer in range(num_layers + 1):
        path = os.path.join(direction_dir, f"direction_layer_{layer:02d}.npz")
        if os.path.exists(path):
            data = np.load(path)
            directions[layer] = {
                "v_unit": data["v_unit"],
                "center": data["center"],
                "v_sentiment": data["v_sentiment"]
            }
    return directions

def extract_single_hidden(model, tokenizer, text, device, max_length=128):
    encoded = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        logits = outputs.logits.squeeze(0).cpu().numpy()
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        hiddens = {l: outputs.hidden_states[l][0, 0, :].cpu().numpy() for l in range(model.config.num_hidden_layers + 1)}
    return hiddens, logits, probs

def generate_ablated_texts(words, indices_to_ablate, edit_type):
    new_words = list(words)
    # 뒤에서부터 지워야 인덱스가 안 꼬임
    for idx in sorted(indices_to_ablate, reverse=True):
        if edit_type == "mask":
            new_words[idx] = "[MASK]"
        elif edit_type == "delete":
            del new_words[idx]
    return " ".join(new_words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--direction-dir", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=str, default="1,2,3")
    parser.add_argument("--random-repeats", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    k_list = [int(x) for x in args.top_k.split(",")]
    
    print(f"[{args.device}] 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(args.device)
    model.eval()
    num_hidden_layers = model.config.num_hidden_layers
    
    directions = load_directions(args.direction_dir, num_hidden_layers)
    if not directions:
        raise ValueError(f"{args.direction_dir} 에 방향성 npz 파일이 없습니다.")
    
    # inputs.txt 읽기
    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
        
    ablation_records = []
    xai_records = []
    
    for sample_id, text in enumerate(tqdm(texts, desc="Ablation 분석")):
        # 1. 원본 추출 및 예측
        h0_dict, logits0, probs0 = extract_single_hidden(model, tokenizer, text, args.device)
        pred0 = np.argmax(probs0)
        prob0 = probs0[pred0]
        y_pred_sign = 1 if pred0 == 1 else -1
        
        # 2. XAI 수행 (IG 기준 수행, Occlusion도 원하면 추가 가능)
        ig_res = explain_integrated_gradients(model, tokenizer, text, device=args.device, steps=50)
        words = ig_res["words"]
        scores = ig_res["scores"]
        
        # XAI 점수 저장
        xai_records.append({
            "sample_id": sample_id,
            "text": text,
            "prediction": pred0,
            "probability": prob0,
            "words": json.dumps(words, ensure_ascii=False),
            "scores": json.dumps(scores)
        })
        
        # 3. 중요도 정렬 (support_score)
        # scores > 0 이면 긍정 지지. pred0가 1이면 그대로, pred0가 0이면 부호를 뒤집음.
        support_scores = np.array(scores) * y_pred_sign
        
        # 인덱스별로 support score 묶어서 정렬
        idx_scores = list(enumerate(support_scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True) # 내림차순 (Top = 가장 지지하는 단어)
        
        for k in k_list:
            if k > len(words): continue
            
            top_k_indices = [x[0] for x in idx_scores[:k]]
            bottom_k_indices = [x[0] for x in idx_scores[-k:]]
            
            controls = {
                "xai_top": [top_k_indices],
                "xai_bottom": [bottom_k_indices],
                "random": []
            }
            
            # Random k 반복
            for _ in range(args.random_repeats):
                rand_idx = np.random.choice(len(words), k, replace=False).tolist()
                controls["random"].append(rand_idx)
                
            for control_type, indices_list in controls.items():
                for iter_idx, indices in enumerate(indices_list):
                    for edit_type in ["mask", "delete"]:
                        modified_text = generate_ablated_texts(words, indices, edit_type)
                        if not modified_text.strip(): continue
                        
                        h1_dict, logits1, probs1 = extract_single_hidden(model, tokenizer, modified_text, args.device)
                        
                        for layer in range(num_hidden_layers + 1):
                            if layer not in directions: continue
                            
                            v_unit = directions[layer]["v_unit"]
                            center = directions[layer]["center"]
                            
                            h0 = h0_dict[layer]
                            h1 = h1_dict[layer]
                            
                            proj0 = np.dot(h0 - center, v_unit)
                            proj1 = np.dot(h1 - center, v_unit)
                            
                            strength0 = proj0 * y_pred_sign
                            strength1 = proj1 * y_pred_sign
                            delta_strength = strength0 - strength1
                            
                            delta_h = h1 - h0
                            delta_h_norm = np.linalg.norm(delta_h) + 1e-12
                            parallel_norm = abs(np.dot(delta_h, v_unit))
                            
                            sentiment_ratio = parallel_norm / delta_h_norm
                            alignment = np.dot(delta_h, -y_pred_sign * v_unit) / delta_h_norm
                            
                            ablation_records.append({
                                "sample_id": sample_id,
                                "text": text,
                                "modified_text": modified_text,
                                "prediction": pred0,
                                "layer": layer,
                                "method": "IG_50",
                                "k": k,
                                "edit_type": edit_type,
                                "control_type": control_type,
                                "iter_idx": iter_idx,
                                "removed_words": json.dumps([words[i] for i in indices], ensure_ascii=False),
                                "projection_original": proj0,
                                "projection_modified": proj1,
                                "delta_projection": proj1 - proj0,
                                "strength_original": strength0,
                                "strength_modified": strength1,
                                "delta_strength": delta_strength,
                                "delta_hidden_norm": delta_h_norm,
                                "parallel_norm": parallel_norm,
                                "sentiment_ratio": sentiment_ratio,
                                "alignment": alignment,
                                "modified_prediction": np.argmax(probs1),
                                "modified_probability": probs1[np.argmax(probs1)]
                            })
                            
    print("Ablation 완료. 통계 요약 중...")
    df_abl = pd.DataFrame(ablation_records)
    df_abl.to_csv(os.path.join(args.output_dir, "xai_sentiment_direction_ablation.csv"), index=False, encoding="utf-8-sig")
    
    # 통계 검정 요약 (Top vs Random, Top vs Bottom for delta_strength)
    summary_records = []
    
    for layer in sorted(df_abl["layer"].unique()):
        for edit_type in ["mask", "delete"]:
            for k in k_list:
                subset = df_abl[(df_abl["layer"] == layer) & (df_abl["edit_type"] == edit_type) & (df_abl["k"] == k)]
                if subset.empty: continue
                
                # mean values
                means = subset.groupby("control_type")[["delta_strength", "alignment", "sentiment_ratio"]].mean()
                
                top_data = subset[subset["control_type"] == "xai_top"].groupby("sample_id")["delta_strength"].mean().values
                rand_data = subset[subset["control_type"] == "random"].groupby("sample_id")["delta_strength"].mean().values
                
                # Wilcoxon
                try:
                    # ensure same size
                    min_len = min(len(top_data), len(rand_data))
                    if min_len > 0:
                        stat, p_val = wilcoxon(top_data[:min_len], rand_data[:min_len], alternative='greater')
                    else:
                        p_val = np.nan
                except ValueError:
                    p_val = np.nan
                    
                summary_records.append({
                    "layer": layer,
                    "edit_type": edit_type,
                    "k": k,
                    "top_delta_strength": means.loc["xai_top", "delta_strength"] if "xai_top" in means.index else np.nan,
                    "rand_delta_strength": means.loc["random", "delta_strength"] if "random" in means.index else np.nan,
                    "wilcoxon_p_val_top_gt_rand": p_val,
                    "top_alignment": means.loc["xai_top", "alignment"] if "xai_top" in means.index else np.nan,
                    "top_sentiment_ratio": means.loc["xai_top", "sentiment_ratio"] if "xai_top" in means.index else np.nan
                })
                
    pd.DataFrame(summary_records).to_csv(os.path.join(args.output_dir, "xai_sentiment_direction_ablation_summary.csv"), index=False, encoding="utf-8-sig")
    
    # XAI 원본 점수
    pd.DataFrame(xai_records).to_csv(os.path.join(args.output_dir, "xai_word_scores_eojeol.csv"), index=False, encoding="utf-8-sig")

    print(f"모든 분석이 완료되었습니다. 결과가 '{args.output_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
