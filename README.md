# Security Finding Severity Classifier

A LoRA-fine-tuned DistilBERT classifier that predicts a security finding's
severity (LOW/MEDIUM/HIGH/CRITICAL) from its title+message text alone —
trained on **real Trivy findings** from scanning this developer's own local
Terraform repos, not synthetic data. Ties the [[lora-sentiment-classifier]]
learning exercise to an actual security-tooling use case (severity triage,
the same shape of problem a scanner product's findings dashboard needs).

## What this proves — and honestly doesn't

| Claim | Evidence |
|---|---|
| The dataset is real, not fabricated | `trivy config` scanned 44+ local Terraform repos (`~/Documents/gcp-projects`, `~/Documents/private`); 882 raw findings, 227 after dedup. See [dataset_provenance.md](dataset_provenance.md). |
| Fine-tuning beats a majority-class baseline, not just an untrained head | Majority-class baseline (always guess LOW, the most common label): **51.1%**. LoRA fine-tuned: **73.3%**. Comparing against majority-class, not just the 44.4% untrained-head number, is the fairer bar on an imbalanced 4-class task. |
| The model isn't secretly only getting the easy classes right | [results.md](results.md)'s confusion matrix: LOW and MEDIUM are classified well (17/23 and 9/10 correct); HIGH is decent (7/11); **CRITICAL is not learned** — only 8 examples exist across the whole dataset, 1 in eval, and it's misclassified as MEDIUM. |
| Small, cheap, laptop-scale fine-tuning | 43.2s training time on Apple M3 (MPS), 1.095% of parameters trained |

**The CRITICAL-class failure is the most important honest result here.**
A severity classifier that can't reliably distinguish CRITICAL from HIGH is
not something you'd wire into an actual alerting pipeline — this dataset
needs either far more CRITICAL examples (scan more repos, or seed
deliberately-critical misconfigurations like [[gcp-security-posture-scanner]]
does) or a different approach (ordinal regression instead of 4-way
classification, since severity is ordered — predicting CRITICAL as HIGH is a
much smaller error than predicting it as LOW, and plain classification
accuracy doesn't capture that).

## Run it

`train.jsonl`/`eval.jsonl` (the extracted title+severity pairs actually used
for training) are committed. The raw `trivy` scan JSON files they were
extracted from are **not** — they're a full dump of file paths and finding
text across 44+ local repos, which is a broader provenance surface than this
demo needs to publish, and one of them turned out to contain a
Stripe-key-shaped string (a deliberately fake test fixture from
`gogenops-cloud-security/fixtures/vulnerable-repo/`, not a live credential —
confirmed and redacted before this was caught a second time by not being
committed at all). `train.jsonl`/`eval.jsonl` themselves never contained it —
`build_dataset.py` only extracts title+message, not the fixture's file
contents.

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch transformers datasets peft accelerate evaluate scikit-learn numpy gradio

python3 train.py    # trains directly from the committed train.jsonl/eval.jsonl, ~1 min
python3 app.py       # Gradio demo at http://127.0.0.1:7860
```

To regenerate the dataset from scratch against your own repos (produces
different numbers, since it reflects your own local Terraform, not this
machine's):

```bash
trivy config --format json --quiet ~/path/to/your/repos > private-scan.json
python3 build_dataset.py   # reads *-scan.json (gitignored), writes train.jsonl/eval.jsonl
```

## Ship it as a free Hugging Face Space

Same pattern as [[lora-sentiment-classifier]] — no paid Inference Endpoint
needed for a model this small:

```bash
huggingface-cli login
huggingface-cli repo create security-severity-demo --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<your-username>/security-severity-demo
cp app.py lora-severity-adapter -r security-severity-demo/
cd security-severity-demo && git add -A && git commit -m "Severity classifier demo" && git push
```

## Repository layout

```
build_dataset.py         trivy scan JSON -> train.jsonl / eval.jsonl (real data, deduped)
train.py                  majority baseline -> untrained-head baseline -> LoRA fine-tune -> results.md
app.py                    Gradio demo
dataset_provenance.md     exact real counts, where the data came from
results.md                real accuracy + confusion matrix from the last training run
lora-severity-adapter/    trained LoRA adapter
*-scan.json                raw trivy output (gitignored — see "Run it" above for why)
```

## Known limits

- **CRITICAL severity is not learned** (see above) — the single biggest honest gap.
- 227 examples total is small for a 4-class task; this is a learning exercise in the LoRA/PEFT workflow, not a production-scale training run.
- The dataset is entirely Terraform/Kubernetes misconfigurations from this one developer's repos — it will not generalize to, say, application-code vulnerability findings (Trivy's other scan types) or another organization's IaC conventions.
- No held-out test set separate from the eval split used to report the final number — same caveat as [[lora-sentiment-classifier]].
