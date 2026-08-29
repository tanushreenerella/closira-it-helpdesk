"""Generate a balanced synthetic IT Helpdesk escalation-intent dataset with Groq.

The output intentionally contains only ``text`` and ``label`` columns so it can
be consumed by the existing baseline and BERT training scripts without changes.

This version writes rows to dataset.csv incrementally and skips work that's
already done, so a rerun after a crash or a daily rate-limit hit RESUMES
instead of starting over and re-burning tokens.
"""

import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TARGET_PER_LABEL = 150
PER_TOPIC_TARGET = 25  # TARGET_PER_LABEL / len(TOPICS)
BATCH_SIZE = 5
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CSV_PATH = Path(__file__).with_name("dataset.csv")
TOPICS = [
    "account access or login",
    "Wi-Fi or network",
    "VPN",
    "device",
    "email",
    "software or access",
]
LABEL_GUIDANCE = {
    "frustrated": (
        "The employee is upset, angry, or strongly dissatisfied with an IT issue. "
        "Do not make an explicit request for a person or human agent."
    ),
    "explicit_human_request": (
        "The employee explicitly asks to speak with, call, chat with, or be transferred "
        "to a human, person, agent, technician, manager, or IT support representative."
    ),
    "repeated_confusion": (
        "The employee says prior explanations, instructions, or attempts did not answer "
        "the question, did not work, or are being repeated."
    ),
    "normal": (
        "A routine, calm IT-support question or status update. Do not include anger, "
        "an explicit human request, or repeated unanswered-attempt language."
    ),
}


def _parse_messages(raw: str) -> list[str]:
    """Extract a JSON array of non-empty messages from a model response.

    Falls back to salvaging complete quoted strings if the array was cut
    off mid-generation (e.g. hit the token limit) and isn't valid JSON.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            values = json.loads(cleaned[start : end + 1])
            if isinstance(values, list):
                return [v.strip() for v in values if isinstance(v, str) and v.strip()]
        except json.JSONDecodeError:
            pass

    if start == -1:
        return []
    tail = cleaned[start:]
    salvaged = re.findall(r'"((?:[^"\\]|\\.)*)"', tail)
    return [json.loads(f'"{s}"').strip() for s in salvaged if s.strip()]


def _prompt(label: str, topic: str, count: int) -> str:
    return f"""Generate exactly {count} distinct synthetic employee messages for an IT Helpdesk intent-classification dataset. This is purely for training a text classifier — no real accounts, credentials, or people are involved.

Target label: {label}
Label definition: {LABEL_GUIDANCE[label]}
IT topic: {topic}

Requirements:
- Each item must be a realistic first-person employee message about the stated IT topic.
- Keep each message under 25 words — short, natural chat messages, not paragraphs.
- Vary wording, tone, urgency, sentence structure, and scenario.
- Do not include actual passwords, one-time codes, secrets, or personally identifying details — describe the *situation*, never real credential values.
- Do not number the messages and do not include labels in the messages.
- Do not include any disclaimer, explanation, or commentary — output only the JSON array itself.
- Return only a valid JSON array of exactly {count} strings.
"""


def _generate_batch(client: Groq, label: str, topic: str, count: int) -> list[str]:
    last_raw = ""
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.9,
                max_tokens=700,
                reasoning_effort="low",
                reasoning_format="hidden",
                messages=[{"role": "user", "content": _prompt(label, topic, count)}],
            )
        except Exception as exc:
            # Surface rate-limit/quota errors immediately instead of
            # burning retries against a daily cap that won't clear soon.
            message = str(exc)
            if "rate_limit_exceeded" in message or "429" in message:
                print(f"\n[generate_dataset] Rate/quota limit hit: {exc}")
                print(
                    "[generate_dataset] Progress so far is already saved in "
                    f"{CSV_PATH.name} — rerun this script later to resume, "
                    "it will skip everything already completed.\n"
                )
                raise SystemExit(1) from exc
            raise

        raw = response.choices[0].message.content or ""
        last_raw = raw
        messages = _parse_messages(raw)
        if len(messages) >= count:
            return messages[:count]

        print(
            f"[generate_dataset] attempt {attempt + 1}/5 for "
            f"label={label!r} topic={topic!r} got {len(messages)}/{count} "
            f"valid messages. Raw response (truncated): {raw[:200]!r}"
        )
        time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"Groq did not return {count} valid {label} messages for {topic} "
        f"after 5 attempts. Last raw response (truncated): {last_raw[:300]!r}"
    )


def _load_existing() -> tuple[list[tuple[str, str]], dict]:
    """Load already-generated rows and per-label counts."""
    if not CSV_PATH.exists():
        return [], defaultdict(int)

    rows: list[tuple[str, str]] = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("text") and row.get("label") in LABEL_GUIDANCE:
                rows.append((row["text"], row["label"]))

    counts = Counter(label for _, label in rows)
    return rows, counts


def _append_rows(rows: list[tuple[str, str]]) -> None:
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["text", "label"])
        writer.writerows(rows)


def generate_dataset() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required to generate the synthetic dataset.")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    existing_rows, label_counts = _load_existing()
    seen = {" ".join(text.lower().split()) for text, _ in existing_rows}

    if existing_rows:
        print(f"[generate_dataset] Resuming — {len(existing_rows)} rows already in {CSV_PATH.name}")
        for label in LABEL_GUIDANCE:
            print(f"  {label}: {label_counts[label]}/{TARGET_PER_LABEL}")

    for label in LABEL_GUIDANCE:
        remaining_for_label = TARGET_PER_LABEL - label_counts[label]
        if remaining_for_label <= 0:
            continue

        per_topic_needed = max(1, remaining_for_label // len(TOPICS))

        for topic in TOPICS:
            if label_counts[label] >= TARGET_PER_LABEL:
                break

            topic_target = min(PER_TOPIC_TARGET, per_topic_needed + 1)
            topic_new: list[str] = []

            while len(topic_new) < topic_target and label_counts[label] < TARGET_PER_LABEL:
                needed = min(BATCH_SIZE, topic_target - len(topic_new))
                batch = _generate_batch(client, label, topic, needed)

                fresh = []
                for message in batch:
                    normalized = " ".join(message.lower().split())
                    if normalized not in seen:
                        seen.add(normalized)
                        fresh.append(message)

                if fresh:
                    _append_rows([(m, label) for m in fresh])
                    topic_new.extend(fresh)
                    label_counts[label] += len(fresh)
                    print(
                        f"[generate_dataset] +{len(fresh)} {label}/{topic} "
                        f"(label total: {label_counts[label]}/{TARGET_PER_LABEL})"
                    )

    print("\n[generate_dataset] Done.")
    for label in LABEL_GUIDANCE:
        print(f"  {label}: {label_counts[label]}/{TARGET_PER_LABEL}")


def main() -> None:
    generate_dataset()


if __name__ == "__main__":
    main()