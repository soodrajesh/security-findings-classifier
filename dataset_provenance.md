# Dataset provenance

Real findings from `trivy config` scans of this machine's own local repos (`~/Documents/gcp-projects`, `~/Documents/private`) — 44+ Terraform projects. No synthetic or fabricated findings; every row is a real Trivy misconfiguration check result against real IaC.

- Raw findings extracted: 882
- After dedup (text, severity): 227
- Train split: 182 · Eval split: 45

Severity distribution (real):

- LOW: 84
- MEDIUM: 71
- HIGH: 64
- CRITICAL: 8
