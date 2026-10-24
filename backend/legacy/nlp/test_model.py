from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path(__file__).parent / "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

    label = model.config.id2label[prediction]
    confidence = probabilities[0][prediction].item()

    return label, confidence


messages = [
    "I am so worried, my Wi-Fi has been down all morning.",
    "Can you please connect me with an IT support representative?",
    "I already followed these instructions twice and they still don't work.",
    "How do I reset my company email password?"
]

for message in messages:
    label, confidence = predict(message)

    print("\nMessage:", message)
    print("Prediction:", label)
    print("Confidence:", round(confidence, 4))