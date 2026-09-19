# backend/nlp/test_classifier.py
from classifier import classify_employee_message

test_cases = [
    ("I am Tanushree working in the development team", "normal"),
    ("This is significantly slowing down my work, I can't stay connected to do my job.", "normal"),
    ("I'm using HP and Windows", "normal"),
    ("How do I reset my email password?", "normal"),
    ("I want to speak to a human", "explicit_human_request"),
    ("This is ridiculous, I'm furious", "frustrated"),
    ("I already tried that, still doesn't work", "repeated_confusion"),
]

for text, expected in test_cases:
    result = classify_employee_message(text)
    match = "✓" if result["label"] == expected else "✗ WRONG"
    print(f"{match} | expected={expected:22s} got={result['label']:22s} conf={result['confidence']:.2f} | {text}")