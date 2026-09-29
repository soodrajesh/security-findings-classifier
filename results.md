# Real results from this run

- Device: mps
- Train examples: 182 · Eval examples: 45
- Majority-class baseline (LOW always): 0.5111
- Pretrained-but-not-fine-tuned accuracy: 0.4444
- LoRA fine-tuned accuracy: 0.7333
- Training wall-clock time: 43.2s
- Trainable params: 741,124 / 67,697,672 (1.095%)

## Confusion matrix (rows=true, cols=predicted)

| | LOW | MEDIUM | HIGH | CRITICAL |
|---|---|---|---|---|
| **LOW** | 17 | 5 | 1 | 0 |
| **MEDIUM** | 1 | 9 | 0 | 0 |
| **HIGH** | 3 | 1 | 7 | 0 |
| **CRITICAL** | 0 | 1 | 0 | 0 |
