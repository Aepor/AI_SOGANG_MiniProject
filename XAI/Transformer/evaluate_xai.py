import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explain_ig import explain_integrated_gradients
from explain_occlusion import explain_occlusion

def plot_attributions(words, scores, title, filename):
    """
    단어와 기여도 점수를 받아, 절댓값이 큰 순서대로 정렬하여 가로 막대 그래프를 그립니다.
    """
    if not words:
        return
        
    # 절댓값 기준으로 내림차순 정렬
    sorted_items = sorted(zip(words, scores), key=lambda x: abs(x[1]), reverse=False) # 가로 막대는 밑에서부터 그려지므로 reverse=False
    sorted_words, sorted_scores = zip(*sorted_items)
    
    # 양수/음수에 따라 색상 지정 (양수: 파란색, 음수: 빨간색)
    colors = ['#ff9999' if s < 0 else '#99ccff' for s in sorted_scores]
    
    plt.figure(figsize=(10, max(4, len(words) * 0.4)))
    y_pos = range(len(sorted_words))
    plt.barh(y_pos, sorted_scores, color=colors)
    plt.yticks(y_pos, sorted_words)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel("Attribution Score (L1 Normalized)", fontsize=12)
    plt.ylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 1. 환경 설정 및 폰트 설정 (Windows 한글 깨짐 방지)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 경로 설정 (실행 위치에 상관없이 절대경로 기반으로 작동)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
    
    # 2. 모델 및 토크나이저 로드
    model_dir = os.path.join(BASE_DIR, "kcelectra_nsmc_model")
    if not os.path.exists(model_dir):
        model_dir = os.path.join(PROJECT_ROOT, "Transformer", "kcelectra_nsmc_model")
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    
    # 3. 입력 텍스트 파일 읽기
    input_file = os.path.join(PROJECT_ROOT, "inputs.txt")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} 가 존재하지 않습니다.")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    # 4. 분석 진행 및 결과 수집 (IG 50단계, 100단계 및 Occlusion)
    results_ig_50 = []
    results_ig_100 = []
    results_occ = []
    
    # 결과 및 그래프를 저장할 폴더 생성
    json_dir = os.path.join(PROJECT_ROOT, "XAI", "outputs_json")
    graph_dir = os.path.join(PROJECT_ROOT, "XAI", "outputs_graph")
    os.makedirs(json_dir, exist_ok=True)
    
    # 모델_기법 폴더 경로 생성
    ig_50_graph_dir = os.path.join(graph_dir, "transformer_ig_50")
    ig_100_graph_dir = os.path.join(graph_dir, "transformer_ig_100")
    occ_graph_dir = os.path.join(graph_dir, "transformer_occlusion")
    
    os.makedirs(ig_50_graph_dir, exist_ok=True)
    os.makedirs(ig_100_graph_dir, exist_ok=True)
    os.makedirs(occ_graph_dir, exist_ok=True)
    
    for i, text in enumerate(lines):
        print(f"[{i+1}/{len(lines)}] 분석 중: {text}")
        
        # IG 분석 (50단계 & 100단계)
        ig_res_50 = explain_integrated_gradients(model, tokenizer, text, device=device, steps=50)
        ig_res_100 = explain_integrated_gradients(model, tokenizer, text, device=device, steps=100)
        
        # Occlusion 분석
        occ_res = explain_occlusion(model, tokenizer, text, device=device)
        
        # 모델의 예측값을 문자열로 변환 (각각 독립적으로 산출)
        pred_str_ig_50 = "positive" if ig_res_50["prediction"] == 1 else "negative"
        pred_str_ig_100 = "positive" if ig_res_100["prediction"] == 1 else "negative"
        pred_str_occ = "positive" if occ_res["prediction"] == 1 else "negative"
        
        # JSON 포맷에 맞게 데이터 결합
        results_ig_50.append({
            "text": text,
            "prediction": pred_str_ig_50,
            "probability": ig_res_50["probability"],
            "words": ig_res_50["words"],
            "scores": ig_res_50["scores"]
        })
        
        results_ig_100.append({
            "text": text,
            "prediction": pred_str_ig_100,
            "probability": ig_res_100["probability"],
            "words": ig_res_100["words"],
            "scores": ig_res_100["scores"]
        })
        
        results_occ.append({
            "text": text,
            "prediction": pred_str_occ,
            "probability": occ_res["probability"],
            "words": occ_res["words"],
            "scores": occ_res["scores"]
        })
        
        # 그래프 시각화 및 저장 (모델_기법 서브폴더 내에 sentence_{번호}.png 로 저장)
        plot_attributions(
            ig_res_50["words"], 
            ig_res_50["scores"], 
            f"Integrated Gradients (Steps 50): {text[:20]}...", 
            os.path.join(ig_50_graph_dir, f"sentence_{i+1}.png")
        )
        plot_attributions(
            ig_res_100["words"], 
            ig_res_100["scores"], 
            f"Integrated Gradients (Steps 100): {text[:20]}...", 
            os.path.join(ig_100_graph_dir, f"sentence_{i+1}.png")
        )
        plot_attributions(
            occ_res["words"], 
            occ_res["scores"], 
            f"Occlusion: {text[:20]}...", 
            os.path.join(occ_graph_dir, f"sentence_{i+1}.png")
        )
        
    # 5. 최종 결과를 각각 분리된 JSON 파일로 저장
    output_file_ig_50 = os.path.join(json_dir, "output_transformer_ig_50.json")
    with open(output_file_ig_50, "w", encoding="utf-8") as f:
        json.dump(results_ig_50, f, ensure_ascii=False, indent=2)
        
    output_file_ig_100 = os.path.join(json_dir, "output_transformer_ig_100.json")
    with open(output_file_ig_100, "w", encoding="utf-8") as f:
        json.dump(results_ig_100, f, ensure_ascii=False, indent=2)
        
    output_file_occ = os.path.join(json_dir, "output_transformer_occlusion.json")
    with open(output_file_occ, "w", encoding="utf-8") as f:
        json.dump(results_occ, f, ensure_ascii=False, indent=2)
        
    print(f"\n[성공] 분석 완료! 결과가 {json_dir}/ 및 {graph_dir}/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
