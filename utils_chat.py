# utils_chat.py
import pandas as pd
import re
from datetime import datetime
import spacy

print("utils_chat imported successfully")

# ---------- spaCy lazy loader ----------

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# WhatsApp line pattern: DD/MM/YY, h:mm am/pm - Name: Message
line_pattern = re.compile(
    r'^(\d{2}/\d{2}/\d{2}),\s+(\d{1,2}:\d{2}\s*[ap]m)\s-\s([^:]+):\s(.*)$'
)

# ---------- parsing uploaded file ----------

def parse_upload_to_df(file_bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8")
    rows = []
    current_msg = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = line_pattern.match(line)
        if m:
            if current_msg is not None:
                rows.append(current_msg)
            current_msg = {
                "date": m.group(1),
                "time": m.group(2),
                "sender": m.group(3).strip(),
                "message": m.group(4).strip(),
            }
        else:
            if current_msg is not None:
                current_msg["message"] += "\n" + line
    if current_msg is not None:
        rows.append(current_msg)
    df = pd.DataFrame(rows)
    return df

# ---------- preprocessing ----------

def clean_message(msg: str) -> str:
    if pd.isna(msg):
        return msg
    patterns_to_drop = [
        r"^<Media omitted>$",
        r"^\u200e?<Media omitted>$",
        r".*joined using this group’s invite link.*",
        r".*changed the subject from.*",
        r".*changed this group’s icon.*",
        r".*added.*",
        r".*left$",
    ]
    for p in patterns_to_drop:
        if re.match(p, msg):
            return ""
    return msg

def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    def parse_dt(row):
        s = f"{row['date']} {row['time']}"
        for fmt in ["%d/%m/%y %I:%M %p", "%d/%m/%y %I:%M %p"]:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return pd.NaT

    df["datetime"] = df.apply(parse_dt, axis=1)
    df["date_only"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["message_clean"] = df["message"].apply(clean_message)
    df = add_datetime_columns(df)
    df_nlp = df[df["message_clean"].str.len() > 0].copy()
    return df_nlp

# ---------- summarization helpers ----------

def summarize_text_safe(text, summarizer,
                        max_input_chars=2000,
                        max_len=80, min_len=20):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = text[-max_input_chars:]
    out = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
    )[0]["summary_text"]
    return out

def build_daily_summaries(df_nlp: pd.DataFrame, summarizer):
    daily_records = []

    MIN_MSGS_PER_DAY = 5
    MAX_MSGS_PER_DAY = 40
    MAX_DAYS = 15

    grouped = list(df_nlp.sort_values("datetime").groupby("date_only"))

    for idx, (day, day_df) in enumerate(grouped):
        if idx >= MAX_DAYS:
            break

        day_msgs = day_df["message_clean"].tolist()
        if len(day_msgs) < MIN_MSGS_PER_DAY:
            continue

        # Only last MAX_MSGS_PER_DAY messages per day for speed
        day_msgs = day_msgs[-MAX_MSGS_PER_DAY:]

        day_text = "\n".join(day_msgs)
        try:
            day_summary = summarize_text_safe(day_text, summarizer)
        except Exception:
            day_summary = ""

        daily_records.append({
            "date": day,
            "num_messages": len(day_msgs),
            "daily_summary": day_summary
        })

    return pd.DataFrame(daily_records).sort_values("date")

# ---------- tasks & deadlines ----------

TASK_KEYWORDS = [
    "register", "registration", "apply", "application",
    "submit", "fill", "complete", "upload",
    "attend", "report", "join",
    "deadline", "last date", "due",
    "make sure", "requested to", "are requested to",
]

DEADLINE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)\b",
    r"\bby\s+\d{1,2}\s*(am|pm|AM|PM)\b",
    r"\bby\s+tomorrow\b",
    r"\bby\s+today\b",
    r"\bTomorrow\b|\btomorrow\b",
    r"\btoday\b|\btonight\b",
    r"\bdeadline\b.*?\b\d{1,2}/\d{1,2}/\d{2,4}\b",
]
deadline_regexes = [re.compile(p, flags=re.IGNORECASE) for p in DEADLINE_PATTERNS]

def is_task_sentence(sent_text: str) -> bool:
    text = sent_text.lower()
    return any(kw in text for kw in TASK_KEYWORDS)

def extract_deadline_text(sent_text: str) -> str:
    for rx in deadline_regexes:
        m = rx.search(sent_text)
        if m:
            return m.group(0)
    doc = get_nlp()(sent_text)
    ents = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    return ", ".join(ents)

def is_strict_deadline(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    if any(w in text_lower for w in ["today", "tomorrow", "pm", "am", "deadline", "by "]):
        return True
    return any(ch.isdigit() for ch in text)

def build_tasks_df(df_nlp: pd.DataFrame) -> pd.DataFrame:
    task_rows = []
    MAX_ROWS = 400   # limit for speed

    for i, (_, row) in enumerate(df_nlp.iterrows()):
        if i >= MAX_ROWS:
            break

        msg = row["message_clean"]
        if not isinstance(msg, str) or len(msg.strip()) == 0:
            continue

        doc = get_nlp()(msg)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not is_task_sentence(sent_text):
                continue
            deadline_text = extract_deadline_text(sent_text)
            task_rows.append({
                "date": row["date_only"],
                "sender": row["sender"],
                "task_sentence": sent_text,
                "deadline_text": deadline_text,
            })

    tasks_df = pd.DataFrame(task_rows)
    if not tasks_df.empty:
        tasks_df["is_strict_deadline"] = tasks_df["deadline_text"].apply(is_strict_deadline)
    return tasks_df
