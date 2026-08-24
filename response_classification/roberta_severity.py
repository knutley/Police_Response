# PART 2: RoBERTa TRAINING AND CLASSIFICATION - Police Severity
# Date: 21-04-2026
# -------------------------------------------------

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch

# ─────────────────────────────────────────────
# PART 1: LOAD AND PREPARE DATA
# ─────────────────────────────────────────────

print("=== LOADING ANNOTATED DATA ===")

pos = pd.read_excel("~/Documents/GitHub/Police_Response/response_pos_set.xlsx")
print(f"Positive set loaded: {len(pos)} rows")

neg = pd.read_excel("~/Documents/GitHub/Police_Response/neg_pos_set.xlsx")
print(f"Negative set loaded: {len(neg)} rows")

LABEL_COLS = ["police_presence", "arrest", "brutality"]

pos = pos[["notes"] + LABEL_COLS]
neg = neg[["notes"] + LABEL_COLS]

# Coerce to numeric and drop any malformed rows
for col in LABEL_COLS:
    pos[col] = pd.to_numeric(pos[col], errors="coerce")
    neg[col] = pd.to_numeric(neg[col], errors="coerce")

pos = pos[pos[LABEL_COLS].isin([0, 1]).all(axis=1)]
neg = neg[neg[LABEL_COLS].isin([0, 1]).all(axis=1)]

print(f"After cleaning - Positive set: {len(pos)} rows, Negative set: {len(neg)} rows")

df = pd.concat([pos, neg], ignore_index=True).fillna(0)
for col in LABEL_COLS:
    df[col] = df[col].astype(int)

print(f"\nCombined dataset: {len(df)} rows")
print("\nLabel distribution:")
for col in LABEL_COLS:
    print(f"  {col.capitalize()}: {df[col].sum()} positive cases ({df[col].mean():.2%})")

# ─────────────────────────────────────────────
# PART 2: TRAIN / TEST SPLIT AND TOKENISATION
# ─────────────────────────────────────────────

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

print("\n=== TOKENIZING DATA ===")
MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(
        batch["notes"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

def prepare_labels(batch):
    batch["labels"] = [
        [float(batch[col][i]) for col in LABEL_COLS]
        for i in range(len(batch[LABEL_COLS[0]]))
    ]
    return batch

train_ds = train_ds.map(prepare_labels, batched=True)
test_ds  = test_ds.map(prepare_labels, batched=True)

train_ds = train_ds.remove_columns(LABEL_COLS)
test_ds  = test_ds.remove_columns(LABEL_COLS)

# ─────────────────────────────────────────────
# PART 3: LOAD MODEL AND DEFINE METRICS
# ─────────────────────────────────────────────

print("\n=== LOADING MODEL ===")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(LABEL_COLS),
    problem_type="multi_label_classification",
)

def compute_metrics(pred):
    logits, labels = pred
    probs  = torch.sigmoid(torch.tensor(logits))
    preds  = (probs > 0.5).int()
    labels = torch.tensor(labels).int()
    return {
        "f1":        f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall":    recall_score(labels, preds, average="micro"),
        "accuracy":  accuracy_score(labels, preds),
    }

# ─────────────────────────────────────────────
# PART 4: TRAIN
# ─────────────────────────────────────────────

print("\n=== TRAINING MODEL ===")
training_args = TrainingArguments(
    output_dir="./police_nlp_model_roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("\n=== FINAL EVALUATION METRICS ===")
print(metrics)

# ─────────────────────────────────────────────
# PART 5: SAVE MODEL
# ─────────────────────────────────────────────

print("\n=== SAVING MODEL ===")
SAVE_DIR = "./police_response_classifier_roberta"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")

# ─────────────────────────────────────────────
# PART 6: CLASSIFY FULL DATASET
# ─────────────────────────────────────────────

print("\n=== CLASSIFYING FULL DATASET ===")

full_data = pd.read_csv(
    "~/Documents/GitHub/Police_Response/data/combined/acled_classified_police_presence1.csv",
    low_memory=False,
)
full_data = full_data.fillna("")
print(f"Loaded {len(full_data)} records to classify")

from transformers import pipeline

clf = pipeline(
    "text-classification",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    top_k=None,
)

THRESHOLD = 0.5

print("Classifying notes (this may take a while)...")
predictions = []
failed = 0

for idx, note in enumerate(full_data["notes"]):
    if idx % 1000 == 0:
        print(f"  Processed {idx}/{len(full_data)} records...")

    note_str = str(note).strip()
    if note_str:
        try:
            raw = clf(note_str[:512])[0]
            # Sort by label name so LABEL_0 → police_presence,
            # LABEL_1 → arrest, LABEL_2 → brutality consistently
            raw_sorted = sorted(raw, key=lambda x: x["label"])

            pred_dict = {}
            for i, col in enumerate(LABEL_COLS):
                score = raw_sorted[i]["score"]
                pred_dict[col]           = int(score > THRESHOLD)  # binary 0/1
                pred_dict[f"{col}_prob"] = round(score, 4)         # raw probability

        except Exception as e:
            print(f"  Warning: Classification failed at index {idx}: {e}")
            pred_dict = {col: 0 for col in LABEL_COLS}
            pred_dict.update({f"{col}_prob": 0.0 for col in LABEL_COLS})
            failed += 1
    else:
        pred_dict = {col: 0 for col in LABEL_COLS}
        pred_dict.update({f"{col}_prob": 0.0 for col in LABEL_COLS})

    predictions.append(pred_dict)

print(f"\nClassification complete. Failed records: {failed}")

pred_df = pd.DataFrame(predictions)

keep_cols = [
    "event_id_cnty", "country", "event_date", "location",
    "notes", "event_partisan_type_final", "police_presence",
]
keep_cols = [c for c in keep_cols if c in full_data.columns]
output_df = pd.concat(
    [full_data[keep_cols].reset_index(drop=True), pred_df],
    axis=1,
)

output_path = "~/Documents/GitHub/Police_Response/data/combined/acled_classified_severity1.csv"
output_df.to_csv(output_path, index=False)

print(f"\n=== CLASSIFICATION COMPLETE ===")
print(f"Results saved to {output_path}")
print(f"\nPrediction summary (at {THRESHOLD} threshold):")
for col in LABEL_COLS:
    n   = pred_df[col].sum()
    pct = pred_df[col].mean()
    print(f"  {col.capitalize()}: {n} cases ({pct:.2%})")