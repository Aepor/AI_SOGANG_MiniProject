import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr

def load_data(train_file, test_file, max_samples=None):
    print("데이터 로딩 중...")
    train_df = pd.read_csv(train_file, sep='\t').dropna()
    test_df = pd.read_csv(test_file, sep='\t').dropna()
    
    # Shuffle and split train -> train/val (90/10)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.iloc[:val_size].copy()
    train_df = train_df.iloc[val_size:].copy()
    
    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples // 10 if max_samples // 10 > 0 else len(val_df))
        test_df = test_df.head(max_samples // 10 if max_samples // 10 > 0 else len(test_df))
        
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def extract_hidden_states(model, tokenizer, texts, device, batch_size=32, max_length=128):
    model.eval()
    
    num_layers = model.config.num_hidden_layers + 1
    all_hidden_states = [[] for _ in range(num_layers)]
    all_logits = []
    all_probs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            for l in range(num_layers):
                cls_hidden = outputs.hidden_states[l][:, 0, :].cpu().numpy()
                all_hidden_states[l].append(cls_hidden)
                
    final_hidden_states = [np.concatenate(layer_states, axis=0) for layer_states in all_hidden_states]
    final_logits = np.concatenate(all_logits, axis=0)
    final_probs = np.concatenate(all_probs, axis=0)
    
    return final_hidden_states, final_logits, final_probs

def get_layer_role(layer, num_hidden_layers):
    if layer == 0:
        return "embedding"
    elif layer <= num_hidden_layers // 3:
        return "early"
    elif layer == num_hidden_layers:
        return "final_reference"
    elif layer in [num_hidden_layers - 4, num_hidden_layers - 3]:
        return "primary_candidate"
    else:
        return "middle"

def calculate_direction(h_train, y_train):
    """v_sentiment_l = mu_pos_l - mu_neg_l"""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    
    mu_pos = np.mean(h_train[pos_idx], axis=0)
    mu_neg = np.mean(h_train[neg_idx], axis=0)
    
    v_sentiment = mu_pos - mu_neg
    norm = np.linalg.norm(v_sentiment)
    v_unit = v_sentiment / (norm + 1e-12)
    center = (mu_pos + mu_neg) / 2.0
    
    return mu_pos, mu_neg, v_sentiment, v_unit, center

def run_linear_probe(h_train, y_train, h_test, y_test, v_sentiment):
    # 속도 최적화: Linear Probe 학습용 데이터를 최대 20,000개로 샘플링 (충분히 수렴함)
    max_train_samples = 20000
    if len(h_train) > max_train_samples:
        h_train_sub = h_train[:max_train_samples]
        y_train_sub = y_train[:max_train_samples]
    else:
        h_train_sub = h_train
        y_train_sub = y_train

    probe = LogisticRegression(max_iter=150, tol=1e-3, random_state=42)
    probe.fit(h_train_sub, y_train_sub)
    
    preds = probe.predict(h_test)
    probs = probe.predict_proba(h_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0)
    }
    
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, probs)
    except ValueError:
        metrics["roc_auc"] = np.nan
        
    w_probe = probe.coef_[0]
    norm_w = np.linalg.norm(w_probe)
    norm_v = np.linalg.norm(v_sentiment)
    if norm_w > 0 and norm_v > 0:
        cosine_sim = np.dot(w_probe, v_sentiment) / (norm_w * norm_v)
    else:
        cosine_sim = 0.0
        
    metrics["cosine_probe_weight_with_v_sentiment"] = cosine_sim
    return metrics, w_probe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None, help="테스트용으로 샘플 수 제한")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[{args.device}] 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(args.device)
    num_hidden_layers = model.config.num_hidden_layers
    
    train_df, val_df, test_df = load_data(args.train_file, args.test_file, max_samples=args.max_samples)
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    hidden_states_dict = {}
    logits_dict = {}
    probs_dict = {}
    
    for split_name, df in splits.items():
        print(f"\n--- {split_name} 분할 처리 중 ---")
        h, l, p = extract_hidden_states(model, tokenizer, df["document"].tolist(), args.device, batch_size=args.batch_size)
        hidden_states_dict[split_name] = h
        logits_dict[split_name] = l
        probs_dict[split_name] = p
        
    projection_records = []
    layer_summary_records = []
    probe_metrics_records = []
    
    print("\n--- 방향성 산출 및 Linear Probe 검증 중 ---")
    for layer in range(num_hidden_layers + 1):
        role = get_layer_role(layer, num_hidden_layers)
        
        # 1. Train 데이터를 이용한 방향성 도출 (Phase 2)
        h_train = hidden_states_dict["train"][layer]
        y_train = splits["train"]["label"].values
        mu_pos, mu_neg, v_sentiment, v_unit, center = calculate_direction(h_train, y_train)
        
        # 방향 정보 npz 저장
        np.savez(
            os.path.join(args.output_dir, f"direction_layer_{layer:02d}.npz"),
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            v_sentiment=v_sentiment,
            v_unit=v_unit,
            center=center
        )
        
        # 2. 전 분할 데이터에 대해 Projection 계산
        layer_proj_pos = []
        layer_proj_neg = []
        all_projections = []
        all_logit_margins = []
        all_y_true = []
        
        for split_name in ["train", "val", "test"]:
            h_split = hidden_states_dict[split_name][layer]
            df_split = splits[split_name]
            logits_split = logits_dict[split_name]
            probs_split = probs_dict[split_name]
            
            # (h - center) dot v_unit (벡터화 연산)
            projections = np.dot(h_split - center, v_unit)
            
            y_true = df_split["label"].values
            preds = np.argmax(probs_split, axis=1)
            probs = probs_split[np.arange(len(preds)), preds]
            logit_margins = logits_split[:, 1] - logits_split[:, 0]
            
            strengths_by_true = np.where(y_true == 1, projections, -projections)
            strengths_by_pred = np.where(preds == 1, projections, -projections)
            
            # pandas iterrows() 루프 제거: DataFrame 생성을 한 번에 수행하여 속도 극대화
            df_layer = pd.DataFrame({
                "sample_id": df_split.index,
                "split": split_name,
                "text": df_split["document"].values,
                "true_label": y_true,
                "prediction": preds,
                "probability": probs,
                "layer": layer,
                "layer_role": role,
                "projection": projections,
                "strength_by_true_label": strengths_by_true,
                "strength_by_prediction": strengths_by_pred,
                "logit_negative": logits_split[:, 0],
                "logit_positive": logits_split[:, 1],
                "logit_margin": logit_margins
            })
            projection_records.append(df_layer)
            
            if split_name == "test": # test 기준으로 summary 생성
                layer_proj_pos.extend(projections[y_true == 1])
                layer_proj_neg.extend(projections[y_true == 0])
                all_projections.extend(projections)
                all_logit_margins.extend(logit_margins)
                all_y_true.extend(y_true)
                    
        # Layer Summary (Test set 기준)
        if len(all_y_true) > 0:
            roc_auc = roc_auc_score(all_y_true, all_projections) if len(set(all_y_true)) > 1 else np.nan
            spearman, _ = spearmanr(all_projections, all_logit_margins)
            pearson, _ = pearsonr(all_projections, all_logit_margins)
            preds_by_proj = (np.array(all_projections) > 0).astype(int)
            proj_acc = accuracy_score(all_y_true, preds_by_proj)
            
            layer_summary_records.append({
                "layer": layer,
                "layer_role": role,
                "n_samples": len(all_y_true),
                "mean_projection_positive": np.mean(layer_proj_pos) if layer_proj_pos else np.nan,
                "mean_projection_negative": np.mean(layer_proj_neg) if layer_proj_neg else np.nan,
                "projection_gap": (np.mean(layer_proj_pos) - np.mean(layer_proj_neg)) if layer_proj_pos and layer_proj_neg else np.nan,
                "threshold_zero_accuracy": proj_acc,
                "roc_auc": roc_auc,
                "spearman_with_logit_margin": spearman,
                "pearson_with_logit_margin": pearson
            })

        # 3. Linear Probe (Phase 3)
        h_test = hidden_states_dict["test"][layer]
        y_test = splits["test"]["label"].values
        
        probe_metrics, w_probe = run_linear_probe(h_train, y_train, h_test, y_test, v_sentiment)
        probe_metrics["layer"] = layer
        probe_metrics["layer_role"] = role
        probe_metrics_records.append(probe_metrics)
        
        np.savez(os.path.join(args.output_dir, f"linear_probe_weights_layer_{layer:02d}.npz"), w_probe=w_probe)
        print(f"Layer {layer:02d} ({role:15s}) - Proj AUC: {roc_auc:.4f}, Probe AUC: {probe_metrics['roc_auc']:.4f}, Alignment: {probe_metrics['cosine_probe_weight_with_v_sentiment']:.4f}")
        
    print("\n결과 CSV 저장 중...")
    pd.concat(projection_records, ignore_index=True).to_csv(os.path.join(args.output_dir, "projection_scores.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(layer_summary_records).to_csv(os.path.join(args.output_dir, "layer_projection_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(probe_metrics_records).to_csv(os.path.join(args.output_dir, "linear_probe_metrics.csv"), index=False, encoding="utf-8-sig")
    
    print(f"분석 완료! 결과가 '{args.output_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
