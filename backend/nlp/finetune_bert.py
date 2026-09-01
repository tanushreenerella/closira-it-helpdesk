"""Fine-tune DistilBERT for Closira's four IT Helpdesk intent labels.

Run from the repository root:
    python backend/nlp/finetune_bert.py

The script writes the trained model and tokenizer to ``backend/nlp/model/``
and records held-out evaluation artifacts alongside the script. It deliberately
does not alter the runtime classifier or agent integration.
"""

import csv
import inspect
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "dataset.csv"
MODEL_DIR = ROOT / "model"
TRAINING_DIR = ROOT / ".training"
METRICS_PATH = ROOT / "evaluation_metrics.json"
REPORT_PATH = ROOT / "classification_report.txt"
MATRIX_PATH = ROOT / "confusion_matrix.csv"
MATRIX_IMAGE_PATH = ROOT / "confusion_matrix.png"
LABELS = ["frustrated", "explicit_human_request", "repeated_confusion", "normal"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
SEED = 42
TRAINABLE_TRANSFORMER_LAYERS = 0


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    expected_columns = {"text", "label"}
    if set(data.columns) != expected_columns:
        raise ValueError(f"Expected CSV columns {expected_columns}; found {list(data.columns)}")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    if (data["text"] == "").any():
        raise ValueError("Dataset contains empty messages.")
    unknown_labels = set(data["label"]) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Dataset contains unsupported labels: {sorted(unknown_labels)}")
    counts = Counter(data["label"])
    if set(counts) != set(LABELS):
        raise ValueError(f"Dataset must contain all labels: {dict(counts)}")
    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, stratified 70/15/15-like splits for each label."""
    smallest_class = min(Counter(data["label"]).values())
    evaluation_per_label = max(1, round(smallest_class * 0.15))
    evaluation_size = evaluation_per_label * len(LABELS)
    if len(data) - evaluation_size < evaluation_size:
        raise ValueError("Dataset is too small to create train, validation, and test splits.")
    train_and_validation, test = train_test_split(
        data, test_size=evaluation_size, random_state=SEED, stratify=data["label"]
    )
    train, validation = train_test_split(
        train_and_validation, test_size=evaluation_size, random_state=SEED, stratify=train_and_validation["label"]
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


class TokenizedMessages(torch.utils.data.Dataset):
    """Minimal PyTorch dataset to avoid requiring the optional datasets package."""

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer):
        self.encodings = tokenizer(
            data["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH
        )
        self.labels = data["label"].map(LABEL_TO_ID).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def to_tokenized_dataset(data: pd.DataFrame, tokenizer: AutoTokenizer) -> TokenizedMessages:
    return TokenizedMessages(data, tokenizer)


def compute_metrics(prediction) -> dict[str, float]:
    logits, label_ids = prediction
    predicted_ids = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_ids, predicted_ids, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(label_ids, predicted_ids),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def configure_trainable_layers(model: AutoModelForSequenceClassification) -> int:
    """Fine-tune the classification head and final encoder layer on limited CPU RAM."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.pre_classifier.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if TRAINABLE_TRANSFORMER_LAYERS:
        for layer in model.distilbert.transformer.layer[-TRAINABLE_TRANSFORMER_LAYERS:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def write_evaluation_artifacts(label_ids: np.ndarray, predicted_ids: np.ndarray, split_sizes: dict[str, int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_ids, predicted_ids, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(label_ids, predicted_ids)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    report = classification_report(label_ids, predicted_ids, labels=list(ID_TO_LABEL), target_names=LABELS, zero_division=0)
    matrix = confusion_matrix(label_ids, predicted_ids, labels=list(ID_TO_LABEL))

    METRICS_PATH.write_text(json.dumps({"split_sizes": split_sizes, "test_metrics": metrics}, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    with MATRIX_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row.tolist()])
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("DistilBERT IT Helpdesk Intent Confusion Matrix")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(MATRIX_IMAGE_PATH, dpi=160)
    plt.close(figure)
    return metrics


def main() -> None:
    set_seed()
    data = load_data()
    train, validation, test = split_data(data)
    split_sizes = {"train": len(train), "validation": len(validation), "test": len(test)}
    print(f"Dataset split: {split_sizes}")
    print(f"Train labels: {dict(sorted(Counter(train['label']).items()))}")
    print(f"Validation labels: {dict(sorted(Counter(validation['label']).items()))}")
    print(f"Test labels: {dict(sorted(Counter(test['label']).items()))}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = to_tokenized_dataset(train, tokenizer)
    validation_dataset = to_tokenized_dataset(validation, tokenizer)
    test_dataset = to_tokenized_dataset(test, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        low_cpu_mem_usage=True,
    )
    trainable_parameters = configure_trainable_layers(model)
    encoder_scope = f"and final {TRAINABLE_TRANSFORMER_LAYERS} DistilBERT layer(s)" if TRAINABLE_TRANSFORMER_LAYERS else "with frozen pretrained encoder"
    print(f"Fine-tuning classifier {encoder_scope}: {trainable_parameters:,} trainable parameters")

    training_kwargs = dict(
        output_dir=str(TRAINING_DIR),
        learning_rate=2e-5,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        optim="adafactor",
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="epoch",
        report_to=[],
        seed=SEED,
    )
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    MODEL_DIR.mkdir(exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(MODEL_DIR)

    prediction = trainer.predict(test_dataset)
    predicted_ids = np.argmax(prediction.predictions, axis=-1)
    metrics = write_evaluation_artifacts(prediction.label_ids, predicted_ids, split_sizes)
    print("Held-out test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Model and tokenizer saved to {MODEL_DIR}")
    print(f"Classification report saved to {REPORT_PATH}")
    print(f"Confusion matrix saved to {MATRIX_PATH}")
    print(f"Confusion matrix image saved to {MATRIX_IMAGE_PATH}")


if __name__ == "__main__":
    main()
