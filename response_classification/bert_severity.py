# PART 2: PYTHON BERT TRAINING AND CLASSIFICATION
# -------------------------------------------------

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch

print("=== LOADING HAND-LABELED TRAINING DATA ===")
# Load your hand-labeled training data (from R script output)
df = pd.read_excel("~/Documents/GitHub/Police_Response/severity_sample.xlsx")
df = df.fillna("")
print(f"Loaded {len(df)} hand-coded samples")
print(f"Columns: {df.columns.tolist()}")

# Keep only text + labels
df = df[["notes", "coercion", "arrest", "brutality"]]

# Convert labels to integers (0 or 1), replacing any NaN with 0
df['coercion'] = df['coercion'].replace('', 0).fillna(0).astype(int)
df['arrest'] = df['arrest'].replace('', 0).fillna(0).astype(int)
df['brutality'] = df['brutality'].replace('', 0).fillna(0).astype(int)

print("\nLabel distribution:")
print(f"Coercion: {df['coercion'].sum()} positive cases ({df['coercion'].mean():.2%})")
print(f"Arrest: {df['arrest'].sum()} positive cases ({df['arrest'].mean():.2%})")
print(f"Brutality: {df['brutality'].sum()} positive cases ({df['brutality'].mean():.2%})")

# Split data into train and test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

print("\n=== TOKENIZING DATA ===")
MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(batch["notes"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Define label columns
label_cols = ["coercion", "arrest", "brutality"]

# Create labels column for multi-label classification
def prepare_labels(batch):
    labels = []
    for i in range(len(batch['coercion'])):
        labels.append([
            float(batch['coercion'][i]),
            float(batch['arrest'][i]),
            float(batch['brutality'][i])
        ])
    batch['labels'] = labels
    return batch

train_ds = train_ds.map(prepare_labels, batched=True)
test_ds = test_ds.map(prepare_labels, batched=True)

# Remove the old columns to avoid confusion
train_ds = train_ds.remove_columns(['coercion', 'arrest', 'brutality'])
test_ds = test_ds.remove_columns(['coercion', 'arrest', 'brutality'])

print("\n=== LOADING MODEL ===")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int()
    labels = torch.tensor(labels)
    return {
        "f1": f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
    }

print("\n=== TRAINING MODEL ===")
training_args = TrainingArguments(
    output_dir="./police_nlp_model",
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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
metrics = trainer.evaluate()
print("\n=== FINAL EVALUATION METRICS ===")
print(metrics)

# Save the trained model
print("\n=== SAVING MODEL ===")
trainer.save_model("./police_response_classifier")
tokenizer.save_pretrained("./police_response_classifier")
print("Model saved to ./police_response_classifier")

# PART 3: CLASSIFY FULL DATASET
# ------------------------------
print("\n=== CLASSIFYING FULL DATASET ===")
full_data = pd.read_csv("~/Downloads/acled_police_response_subset.csv")
full_data = full_data.fillna("")
print(f"Loaded {len(full_data)} records to classify")

# Load the trained model
clf = pipeline("text-classification", 
               model="./police_response_classifier", 
               tokenizer="./police_response_classifier", 
               return_all_scores=True)

# Classify all notes
print("Classifying notes (this may take a while)...")
predictions = []
for idx, note in enumerate(full_data['notes']):
    if idx % 100 == 0:
        print(f"  Processed {idx}/{len(full_data)} records...")
    
    if note and str(note).strip():  # Only if there's text
        try:
            result = clf(str(note)[:512])[0]  # Truncate long texts
            pred_dict = {label_cols[i]: result[i]['score'] for i in range(len(label_cols))}
        except:
            pred_dict = {col: 0 for col in label_cols}
    else:
        pred_dict = {col: 0 for col in label_cols}
    predictions.append(pred_dict)

# Add predictions to dataframe
pred_df = pd.DataFrame(predictions)
full_data_with_preds = pd.concat([full_data, pred_df], axis=1)

# Save results
output_path = "~/Downloads/acled_classified_results.csv"
full_data_with_preds.to_csv(output_path, index=False)
print(f"\n=== CLASSIFICATION COMPLETE ===")
print(f"Results saved to {output_path}")
print(f"\nPrediction summary:")
print(f"Coercion (>0.5): {(pred_df['coercion'] > 0.5).sum()} cases ({(pred_df['coercion'] > 0.5).mean():.2%})")
print(f"Arrest (>0.5): {(pred_df['arrest'] > 0.5).sum()} cases ({(pred_df['arrest'] > 0.5).mean():.2%})")
print(f"Brutality (>0.5): {(pred_df['brutality'] > 0.5).sum()} cases ({(pred_df['brutality'] > 0.5).mean():.2%})")