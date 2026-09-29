"""
Build a (finding text -> severity) dataset from real trivy config scans of
this machine's own local repos. No synthetic/fabricated findings — every
row comes from Trivy actually evaluating real Terraform against real rules.
"""
import json
import random

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def extract(path):
    data = json.load(open(path))
    rows = []
    for result in data.get("Results") or []:
        target = result.get("Target", "")
        for m in result.get("Misconfigurations") or []:
            title = m.get("Title", "")
            desc = m.get("Description", "")
            message = m.get("Message", "")
            severity = m.get("Severity", "")
            if severity not in SEVERITIES:
                continue
            # Use title + message as the input text — this is what a triage
            # tool would actually have to work with (not the full description,
            # which often just repeats the title).
            text = f"{title}. {message}".strip()
            if len(text) < 10:
                text = f"{title}. {desc}".strip()
            rows.append({"text": text, "severity": severity, "target": target})
    return rows


rows = extract("gcp-projects-scan.json") + extract("private-scan.json")
print(f"Total raw rows: {len(rows)}")

# Dedupe by (text, severity) — the same rule fires on many near-identical
# resources across repos, and we don't want the eval split to just be
# memorized duplicates of the train split.
seen = set()
deduped = []
for r in rows:
    key = (r["text"], r["severity"])
    if key in seen:
        continue
    seen.add(key)
    deduped.append(r)

print(f"After dedup: {len(deduped)}")

from collections import Counter

counts = Counter(r["severity"] for r in deduped)
print("Severity distribution (real, from actual Trivy findings):")
for sev in SEVERITIES:
    print(f"  {sev}: {counts.get(sev, 0)}")

random.seed(42)
random.shuffle(deduped)

n_eval = max(20, int(len(deduped) * 0.2))
eval_rows = deduped[:n_eval]
train_rows = deduped[n_eval:]

print(f"\nTrain: {len(train_rows)} · Eval: {len(eval_rows)}")

with open("train.jsonl", "w") as f:
    for r in train_rows:
        f.write(json.dumps({"text": r["text"], "severity": r["severity"]}) + "\n")

with open("eval.jsonl", "w") as f:
    for r in eval_rows:
        f.write(json.dumps({"text": r["text"], "severity": r["severity"]}) + "\n")

with open("dataset_provenance.md", "w") as f:
    f.write("# Dataset provenance\n\n")
    f.write(
        "Real findings from `trivy config` scans of this machine's own local "
        "repos (`~/Documents/gcp-projects`, `~/Documents/private`) — 44+ "
        "Terraform projects. No synthetic or fabricated findings; every row "
        "is a real Trivy misconfiguration check result against real IaC.\n\n"
    )
    f.write(f"- Raw findings extracted: {len(rows)}\n")
    f.write(f"- After dedup (text, severity): {len(deduped)}\n")
    f.write(f"- Train split: {len(train_rows)} · Eval split: {len(eval_rows)}\n\n")
    f.write("Severity distribution (real):\n\n")
    for sev in SEVERITIES:
        f.write(f"- {sev}: {counts.get(sev, 0)}\n")

print("\nWrote train.jsonl, eval.jsonl, dataset_provenance.md")
