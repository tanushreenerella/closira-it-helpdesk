"""Fine-tune DistilBERT and save it to nlp/bert_model/."""

from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

root = Path(__file__).parent
labels = ["frustrated", "explicit_human_request", "repeated_confusion", "normal"]
label_to_id = {label: index for index, label in enumerate(labels)}
data = pd.read_csv(root / "dataset.csv")
dataset = Dataset.from_pandas(data.assign(labels=data.label.map(label_to_id))[["text", "labels"]])
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128), batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label_to_id)
args = TrainingArguments(output_dir=str(root / ".training"), num_train_epochs=3, per_device_train_batch_size=8, save_strategy="no", report_to=[])
Trainer(model=model, args=args, train_dataset=dataset).train()
model.save_pretrained(root / "bert_model")
tokenizer.save_pretrained(root / "bert_model")
