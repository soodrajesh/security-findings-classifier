"""
Gradio demo: predicts a security finding's severity (LOW/MEDIUM/HIGH/CRITICAL)
from its title+message text, using the LoRA adapter trained by train.py on
real Trivy findings from this machine's own repos.
"""
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "distilbert-base-uncased"
ADAPTER_DIR = "./lora-severity-adapter"
LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)
model.eval()


def classify(text: str):
    if not text.strip():
        return {}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {LABELS[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="Finding title + message",
        placeholder="Instance with unencrypted block device. EBS block device is not encrypted.",
        lines=3,
    ),
    outputs=gr.Label(label="Predicted severity"),
    title="Security Finding Severity Classifier",
    description=(
        "DistilBERT + LoRA, fine-tuned on 182 REAL Trivy findings from scanning "
        "this developer's own Terraform repos (44+ local projects). Real eval "
        "accuracy: 51.1% majority-class baseline -> 73.3% after fine-tuning. "
        "CRITICAL is under-represented (8 examples total) — see results.md and "
        "dataset_provenance.md for the honest numbers, including the confusion "
        "matrix. This is a learning exercise, not a production triage tool."
    ),
    examples=[
        "Instance with unencrypted block device. EBS block device is not encrypted.",
        "Default capabilities: some containers do not drop all. Container should add 'ALL' to droped capabilities.",
        "Cloud Storage Bucket Versioning Disabled. Storage bucket versioning is not enabled.",
        "Security group allows ingress from 0.0.0.0/0 to port 22. SSH is open to the entire internet.",
    ],
)

if __name__ == "__main__":
    demo.launch()
