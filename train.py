"""
LoRA fine-tune distilbert-base-uncased to predict a finding's severity
(LOW/MEDIUM/HIGH/CRITICAL) from its title+message text alone — trained on
real Trivy findings from this machine's own repos (see build_dataset.py /
dataset_provenance.md). Same pattern as ../lora-sentiment-classifier/train.py,
extended to 4 classes and a much smaller, real (not synthetic) dataset.
"""
import json
import time

import evaluate
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    return Dataset.from_dict(
        {
            "text": [r["text"] for r in rows],
            "labels": [LABEL2ID[r["severity"]] for r in rows],
        }
    )


train_ds = load_jsonl("train.jsonl")
eval_ds = load_jsonl("eval.jsonl")
print(f"train: {len(train_ds)} · eval: {len(eval_ds)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)


train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)
cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=cols)
eval_ds.set_format(type="torch", columns=cols)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)


def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)


# --- Baselines ------------------------------------------------------------
train_label_counts = np.bincount([train_ds[i]["labels"].item() for i in range(len(train_ds))], minlength=4)
majority_label = int(np.argmax(train_label_counts))
eval_labels = np.array([eval_ds[i]["labels"].item() for i in range(len(eval_ds))])
majority_baseline_acc = float((eval_labels == majority_label).mean())
print(f"\nMajority-class baseline (always predict {LABELS[majority_label]}): {majority_baseline_acc:.4f}")

print("\n=== Baseline: pretrained head, not fine-tuned ===")
baseline_model = build_model()
args0 = TrainingArguments(
    output_dir="./tmp-eval-baseline", per_device_eval_batch_size=32, report_to=[], use_cpu=(device == "cpu")
)
baseline_trainer = Trainer(model=baseline_model, args=args0, eval_dataset=eval_ds, compute_metrics=compute_metrics)
baseline_metrics = baseline_trainer.evaluate()
print(baseline_metrics)

# --- LoRA fine-tune ---------------------------------------------------------
print("\n=== LoRA fine-tuning ===")
base_model = build_model()
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
)
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora-severity-checkpoints",
    num_train_epochs=8,  # small dataset -> more epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=3e-4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    report_to=[],
    use_cpu=(device == "cpu"),
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"\nTraining took {elapsed:.1f}s on {device}")

final_metrics = trainer.evaluate()
print("\n=== Final LoRA-fine-tuned accuracy ===")
print(final_metrics)

# Confusion matrix — this is the honest way to show a 4-class, imbalanced,
# 45-example eval set isn't secretly just getting the majority class right.
preds = trainer.predict(eval_ds).predictions.argmax(axis=-1)
cm = confusion_matrix(eval_labels, preds, labels=list(range(len(LABELS))))

lora_model.save_pretrained("./lora-severity-adapter")
tokenizer.save_pretrained("./lora-severity-adapter")

with open("results.md", "w") as f:
    f.write("# Real results from this run\n\n")
    f.write(f"- Device: {device}\n")
    f.write(f"- Train examples: {len(train_ds)} · Eval examples: {len(eval_ds)}\n")
    f.write(f"- Majority-class baseline ({LABELS[majority_label]} always): {majority_baseline_acc:.4f}\n")
    f.write(f"- Pretrained-but-not-fine-tuned accuracy: {baseline_metrics['eval_accuracy']:.4f}\n")
    f.write(f"- LoRA fine-tuned accuracy: {final_metrics['eval_accuracy']:.4f}\n")
    f.write(f"- Training wall-clock time: {elapsed:.1f}s\n")
    trainable, total = lora_model.get_nb_trainable_parameters()
    f.write(f"- Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)\n\n")
    f.write("## Confusion matrix (rows=true, cols=predicted)\n\n")
    f.write("| | " + " | ".join(LABELS) + " |\n")
    f.write("|---|" + "---|" * len(LABELS) + "\n")
    for i, row_label in enumerate(LABELS):
        f.write(f"| **{row_label}** | " + " | ".join(str(x) for x in cm[i]) + " |\n")

print("\nWrote results.md")
