import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: transformers. Install dependencies with "
        "`pip install -r XAI/Transformer/requirements.txt`."
    ) from exc


LABELS = {0: "negative", 1: "positive"}


class NSMCDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """NSMC 문장과 라벨을 tokenizer로 모델 입력 형태로 바꿀 준비를 합니다."""
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """데이터셋에 들어 있는 전체 문장 개수를 반환합니다."""
        return len(self.texts)

    def __getitem__(self, index):
        """하나의 문장을 토큰화하고, 라벨이 있으면 함께 반환합니다."""
        encoded = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def set_seed(seed):
    """실험 재현성을 위해 Python과 PyTorch의 랜덤 시드를 고정합니다."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    """사용자가 지정한 장치 또는 사용 가능한 기본 장치를 선택합니다."""
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_nsmc(path, sample_size=None, seed=42):
    """NSMC TSV 파일을 읽고 학습에 필요한 document와 label 컬럼을 정리합니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NSMC file not found: {path}")

    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            frame = pd.read_csv(path, sep="\t", encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise UnicodeDecodeError(
            "unknown", b"", 0, 1, f"Could not decode {path}: {last_error}"
        )

    expected_columns = {"document", "label"}
    if not expected_columns.issubset(set(frame.columns)):
        raise ValueError(f"{path} must contain columns: id, document, label")

    frame = frame.dropna(subset=["document", "label"]).copy()
    frame["document"] = frame["document"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if sample_size:
        frame = frame.sample(n=min(sample_size, len(frame)), random_state=seed)
    return frame.reset_index(drop=True)


def split_train_validation(frame, validation_ratio, seed=42):
    """학습 데이터를 train과 validation으로 나누고 라벨 비율을 최대한 유지합니다."""
    if not 0 < validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1.")

    valid_frame = frame.groupby("label", group_keys=False).sample(
        frac=validation_ratio,
        random_state=seed,
    )
    train_frame = frame.drop(valid_frame.index)

    if len(train_frame) == 0 or len(valid_frame) == 0:
        raise ValueError("Train/validation split produced an empty split.")

    train_frame = train_frame.sample(frac=1, random_state=seed)
    valid_frame = valid_frame.sample(frac=1, random_state=seed)
    return train_frame.reset_index(drop=True), valid_frame.reset_index(drop=True)


def build_loader(frame, tokenizer, max_length, batch_size, shuffle):
    """정리된 NSMC DataFrame을 PyTorch DataLoader로 변환합니다."""
    dataset = NSMCDataset(
        texts=frame["document"],
        labels=frame["label"],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def serializable_args(args):
    """JSON으로 저장할 수 없는 argparse 내부 함수 값을 제외하고 인자를 정리합니다."""
    return {
        key: value
        for key, value in vars(args).items()
        if not callable(value)
    }


def evaluate(model, loader, device):
    """평가 데이터로 평균 loss, accuracy, precision, recall, F1을 계산합니다."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct_count = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            predictions = logits.argmax(dim=-1)
            batch_size = labels.size(0)

            total_loss += outputs.loss.item() * batch_size
            total_count += batch_size
            correct_count += (predictions == labels).sum().item()
            true_positive += ((predictions == 1) & (labels == 1)).sum().item()
            false_positive += ((predictions == 1) & (labels == 0)).sum().item()
            false_negative += ((predictions == 0) & (labels == 1)).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": correct_count / max(total_count, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train(args):
    """KcELECTRA 모델을 NSMC 데이터로 fine-tuning하고 최고 성능 체크포인트를 저장합니다."""
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=LABELS,
        label2id={value: key for key, value in LABELS.items()},
    ).to(device)

    full_train_frame = read_nsmc(args.train_file, args.train_sample, args.seed)
    if args.valid_file:
        train_frame = full_train_frame
        valid_frame = read_nsmc(args.valid_file, args.valid_sample, args.seed)
    else:
        train_frame, valid_frame = split_train_validation(
            full_train_frame, args.validation_ratio, args.seed
        )
        if args.valid_sample:
            valid_frame = valid_frame.sample(
                n=min(args.valid_sample, len(valid_frame)),
                random_state=args.seed,
            ).reset_index(drop=True)

    print(f"train_size={len(train_frame)} valid_size={len(valid_frame)}")
    train_loader = build_loader(
        train_frame, tokenizer, args.max_length, args.batch_size, shuffle=True
    )
    valid_loader = build_loader(
        valid_frame, tokenizer, args.max_length, args.eval_batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps *= args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_loss = float("inf")
    history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}")

        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps

            should_step = step % args.gradient_accumulation_steps == 0
            is_last_step = step == len(train_loader)
            if should_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        metrics = evaluate(model, valid_loader, device)
        metrics["epoch"] = epoch
        metrics["split"] = "validation"
        history.append(metrics)
        print(
            f"epoch={epoch} valid_loss={metrics['loss']:.4f} "
            f"valid_accuracy={metrics['accuracy']:.4f} "
            f"valid_precision={metrics['precision']:.4f} "
            f"valid_recall={metrics['recall']:.4f} "
            f"valid_f1={metrics['f1']:.4f}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "training_args.json").write_text(
                json.dumps(serializable_args(args), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"saved best model to {output_dir} (valid_loss={best_loss:.4f})")


def load_model_and_tokenizer(model_dir, device):
    """저장된 폴더에서 fine-tuning된 모델과 같은 tokenizer를 불러옵니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def predict_texts(args):
    """입력 문장마다 부정/긍정 확률과 최종 예측 라벨을 출력합니다."""
    device = get_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    texts = args.text or []
    if args.input_file:
        texts.extend(
            line.strip()
            for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    if not texts:
        raise ValueError("Provide --text or --input-file.")

    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1).squeeze(0)
        pred_id = int(probs.argmax().item())
        print(
            json.dumps(
                {
                    "text": text,
                    "label": LABELS[pred_id],
                    "score": float(probs[pred_id].item()),
                    "negative": float(probs[0].item()),
                    "positive": float(probs[1].item()),
                },
                ensure_ascii=False,
            )
        )


def evaluate_saved(args):
    """저장된 모델을 불러와 NSMC 테스트 데이터에서 성능을 평가합니다."""
    device = get_device(args.device)
    model_dir = Path(args.model_dir)
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    test_frame = read_nsmc(args.test_file, args.eval_sample, args.seed)
    eval_loader = build_loader(
        test_frame, tokenizer, args.max_length, args.eval_batch_size, shuffle=False
    )
    metrics = evaluate(model, eval_loader, device)
    metrics["split"] = "test"
    metrics["test_file"] = args.test_file
    print("Loss:", metrics["loss"])
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])

    metrics_path = model_dir / "metrics.json"
    history = []
    if metrics_path.exists():
        history = json.loads(metrics_path.read_text(encoding="utf-8"))
    history.append(metrics)
    metrics_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved test metrics to {metrics_path}")


def add_common_runtime_args(parser):
    """여러 CLI 명령에서 공통으로 쓰는 max_length와 device 인자를 추가합니다."""
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")


def parse_args():
    """train, eval, predict 서브커맨드와 각 옵션을 정의합니다."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate beomi/KcELECTRA-base on NSMC sentiment data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model-name", default="beomi/KcELECTRA-base")
    train_parser.add_argument("--train-file", default="Data/NSMC/ratings_train.txt")
    train_parser.add_argument("--valid-file")
    train_parser.add_argument("--validation-ratio", type=float, default=0.1)
    train_parser.add_argument("--test-file", help=argparse.SUPPRESS)
    train_parser.add_argument("--output-dir", default="XAI/Transformer/kcelectra_nsmc_model")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--train-sample", type=int)
    train_parser.add_argument("--valid-sample", type=int)
    add_common_runtime_args(train_parser)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model-dir", default="XAI/Transformer/kcelectra_nsmc_model")
    eval_parser.add_argument("--test-file", default="Data/NSMC/ratings_test.txt")
    eval_parser.add_argument("--eval-batch-size", type=int, default=32)
    eval_parser.add_argument("--eval-sample", type=int)
    eval_parser.add_argument("--seed", type=int, default=42)
    add_common_runtime_args(eval_parser)
    eval_parser.set_defaults(func=evaluate_saved)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-dir", default="XAI/Transformer/kcelectra_nsmc_model")
    predict_parser.add_argument("--text", action="append")
    predict_parser.add_argument("--input-file")
    add_common_runtime_args(predict_parser)
    predict_parser.set_defaults(func=predict_texts)

    return parser.parse_args()


def main():
    """CLI 인자를 해석한 뒤 선택된 서브커맨드 함수를 실행합니다."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
