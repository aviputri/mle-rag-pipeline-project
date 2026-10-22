# Calibration Results

- Questions: `data/evaluation/calibration.json` (6 cases)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- k = 5, collection `northstar_kb`
- Retrieval (supported cases): recall@5 = 1.0, mrr = 0.833, evidence_recall@5 = 1.0

## Per-question signals

| id | expected | retrieved (chunk: score) | top | margin | doc rank | evidence rank | term cov. | query overlap |
|---|---|---|---|---|---|---|---|---|
| cal01 | billing-refunds | invoice-corrections#1: 0.550, billing-refunds#2: 0.533, billing-refunds#0: 0.446, billing-refunds#1: 0.423, invoice-corrections#0: 0.338 | 0.550 | 0.017 | 2 | 3 | 1.00 | 0.83 |
| cal02 | account-mfa | account-mfa#0: 0.496, account-mfa#1: 0.364, account-password#1: 0.341, account-password#0: 0.324, invoice-corrections#0: 0.284 | 0.496 | 0.131 | 1 | 1 | 1.00 | 1.00 |
| cal03 | technical-api-limits | technical-api-limits#0: 0.708, billing-refunds#0: 0.392, technical-incidents#1: 0.228, technical-incidents#0: 0.181, programme-extensions#1: 0.180 | 0.708 | 0.316 | 1 | 1 | 1.00 | 0.67 |
| cal04 | ABSTAIN | programme-extensions#1: 0.310, account-password#1: 0.292, technical-incidents#0: 0.284, programme-extensions#0: 0.269, account-mfa#0: 0.264 | 0.310 | 0.018 | - | - | - | 0.40 |
| cal05 | ABSTAIN | billing-refunds#2: 0.441, billing-refunds#0: 0.392, invoice-corrections#1: 0.361, billing-refunds#1: 0.347, account-password#1: 0.307 | 0.441 | 0.049 | - | - | - | 0.60 |
| cal06 | ABSTAIN | billing-refunds#2: 0.293, technical-api-limits#0: 0.252, invoice-corrections#1: 0.213, account-password#0: 0.209, programme-attendance#1: 0.174 | 0.294 | 0.042 | - | - | - | 0.50 |

## Score separation

- Lowest supported top score: 0.496
- Highest unsupported top score: 0.441
- Separable by score alone: True (midpoint 0.468)

## Rule A: score threshold

| min_score | false_answers | false_refusals | decision_accuracy |
|---|---|---|---|
| 0.20 | 3 | 0 | 0.50 |
| 0.25 | 3 | 0 | 0.50 |
| 0.30 | 2 | 0 | 0.67 |
| 0.35 | 1 | 0 | 0.83 |
| 0.40 | 1 | 0 | 0.83 |
| 0.45 | 0 | 0 | 1.00 |
| 0.50 | 0 | 1 | 0.83 |
| 0.55 | 0 | 2 | 0.67 |
| 0.60 | 0 | 2 | 0.67 |
| 0.65 | 0 | 2 | 0.67 |
| 0.70 | 0 | 2 | 0.67 |

## Best settings per rule

### A_score

| min_score | false_answers | false_refusals | decision_accuracy |
|---|---|---|---|
| 0.45 | 0 | 0 | 1.00 |

### B_score_margin

| min_score | min_margin | false_answers | false_refusals | decision_accuracy |
|---|---|---|---|---|
| 0.45 | 0.00 | 0 | 0 | 1.00 |
| 0.45 | 0.01 | 0 | 0 | 1.00 |

### C_score_overlap

| min_score | min_overlap | false_answers | false_refusals | decision_accuracy |
|---|---|---|---|---|
| 0.45 | 0.00 | 0 | 0 | 1.00 |
| 0.45 | 0.20 | 0 | 0 | 1.00 |
| 0.45 | 0.30 | 0 | 0 | 1.00 |
| 0.45 | 0.40 | 0 | 0 | 1.00 |
| 0.45 | 0.50 | 0 | 0 | 1.00 |
| 0.45 | 0.60 | 0 | 0 | 1.00 |

> With only 6 calibration cases, many settings tie. Prefer the simplest
> rule, and a threshold away from the observed scores, over a value that
> merely fits these six questions.
