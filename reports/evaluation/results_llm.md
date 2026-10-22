# Evaluation Results (llm:groq/openai/gpt-oss-20b)

- Questions: `data/evaluation/questions.json` (16 cases: 13 supported, 3 should abstain)
- Generation mode: `llm:groq/openai/gpt-oss-20b`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- TOP_K = 3, MIN_SCORE = 0.47 (frozen after calibration)
- System prompt sha1: `f446011993`, git commit: `ade5597+uncommitted`
- Run at 2026-10-22 13:51 UTC (13.6 s)

## Summary

| Metric | Value |
|---|---|
| hit@k | 1.000 |
| recall@k | 1.000 |
| mrr | 0.923 |
| decision_accuracy | 0.875 |
| false_answers | 0 |
| false_refusals | 2 |
| answer_relevance | 0.577 |
| faithfulness | 0.773 |
| faithful_answers | 8 |
| answers_needing_review | 11 |
| citation_precision | 1.000 |

Status counts: {'answered': 11, 'abstained': 5, 'blocked_input': 0, 'citation_failed': 0, 'error': 0}

## Per-question results

| id | expected | top-3 retrieved (score) | top | status | decision | first rank | relevance | faithfulness | cited |
|---|---|---|---|---|---|---|---|---|---|
| q01 | billing-refunds | invoice-corrections#1 (0.634), billing-refunds#2 (0.505), billing-refunds#0 (0.470) | 0.634 | answered | ok | 2 | 1.00 | 1.00 (review) | billing-refunds |
| q02 | billing-refunds | billing-refunds#1 (0.756), billing-refunds#0 (0.534), billing-refunds#2 (0.522) | 0.756 | answered | ok | 1 | 1.00 | 1.00 (review) | billing-refunds |
| q03 | invoice-corrections | invoice-corrections#0 (0.762), invoice-corrections#1 (0.585), billing-refunds#2 (0.434) | 0.762 | answered | ok | 1 | 0.50 | 1.00 (review) | invoice-corrections |
| q04 | account-mfa | account-mfa#0 (0.623), account-mfa#1 (0.491), account-password#1 (0.283) | 0.623 | answered | ok | 1 | 0.50 | 1.00 (review) | account-mfa |
| q05 | account-password | account-password#0 (0.821), account-password#1 (0.692), account-mfa#0 (0.331) | 0.821 | answered | ok | 1 | 1.00 | 1.00 (review) | account-password |
| q06 | programme-extensions | programme-extensions#0 (0.518), programme-extensions#1 (0.416), programme-attendance#0 (0.329) | 0.518 | answered | ok | 1 | 1.00 | 1.00 (review) | programme-extensions |
| q07 | technical-api-limits | technical-incidents#1 (0.341), technical-api-limits#0 (0.305), retrieved-content-safety#2 (0.302) | 0.341 | abstained | WRONG | 2 | 0.00 | - | - |
| q08 | technical-incidents | technical-incidents#0 (0.680), technical-incidents#1 (0.424), programme-attendance#0 (0.248) | 0.680 | answered | ok | 1 | 0.00 | 0.00 (review) | technical-incidents |
| q09 | privacy-data, technical-incidents | technical-incidents#1 (0.347), privacy-data#0 (0.328), account-mfa#1 (0.303) | 0.347 | abstained | WRONG | 1 | 0.00 | - | - |
| q10 | retrieved-content-safety | retrieved-content-safety#0 (0.620), retrieved-content-safety#2 (0.487), privacy-data#0 (0.414) | 0.620 | answered | ok | 1 | 0.50 | 1.00 (review) | retrieved-content-safety |
| q11 | ABSTAIN | programme-attendance#0 (0.168), programme-attendance#1 (0.131), technical-incidents#0 (0.100) | 0.168 | abstained | ok | - | - | - | - |
| q12 | ABSTAIN | technical-api-limits#0 (0.212), invoice-corrections#1 (0.157), billing-refunds#2 (0.140) | 0.212 | abstained | ok | - | - | - | - |
| q13 | account-mfa, privacy-data | account-mfa#1 (0.557), account-mfa#0 (0.391), privacy-data#1 (0.320) | 0.557 | answered | ok | 1 | 0.50 | 0.50 (review) | account-mfa |
| q14 | programme-extensions | programme-extensions#1 (0.771), programme-extensions#0 (0.623), technical-incidents#0 (0.287) | 0.771 | answered | ok | 1 | 0.50 | 0.00 (review) | programme-extensions |
| q15 | programme-attendance | programme-attendance#0 (0.590), programme-attendance#1 (0.452), privacy-data#1 (0.289) | 0.590 | answered | ok | 1 | 1.00 | 1.00 (review) | programme-attendance |
| q16 | ABSTAIN | programme-attendance#1 (0.110), privacy-data#0 (0.086), programme-attendance#0 (0.076) | 0.110 | abstained | ok | - | - | - | - |

## Answers and failure details

### q01: I was charged twice for the same invoice. What must I send and when?

- status: `answered` (passed threshold and citation checks)
- expected: ['billing-refunds']; required terms: ['invoice number', 'payment references', '30']

> Send the invoice number and both payment references within 30 calendar days of the second charge. [billing-refunds]

- supported (word support 0.909): Send the invoice number and both payment references within 30 calendar days of the second charge [billing-refunds].

### q02: How long does finance need to review a complete duplicate-payment request?

- status: `answered` (passed threshold and citation checks)
- expected: ['billing-refunds']; required terms: ['three business days']

> Finance reviews a complete duplicate‑payment request within three business days. [billing-refunds]

- supported (word support 1.0): Finance reviews a complete duplicate‑payment request within three business days [billing-refunds].

### q03: Can the legal company name on an invoice be corrected?  <-- inspect

- status: `answered` (passed threshold and citation checks)
- expected: ['invoice-corrections']; required terms: ['legal company name', 'invoice number']

> Yes, finance can correct the legal company name on an issued invoice. [invoice-corrections]

- supported (word support 0.875): Yes, finance can correct the legal company name on an issued invoice [invoice-corrections].

### q04: I replaced my phone and lost my authenticator. What should I do?  <-- inspect

- status: `answered` (passed threshold and citation checks)
- expected: ['account-mfa']; required terms: ['identity verification', 'registered email']

> Open a support ticket from your registered email and describe the lost‑device situation. [account-mfa] After verification, the previous authenticator will be disabled and you can enroll a new device at the next sign‑in. [account-mfa]

- supported (word support 1.0): Open a support ticket from your registered email and describe the lost‑device situation [account-mfa].
- supported (word support 0.9): After verification, the previous authenticator will be disabled and you can enroll a new device at the next sign‑in [account-mfa].

### q05: How quickly does a password reset link expire?

- status: `answered` (passed threshold and citation checks)
- expected: ['account-password']; required terms: ['20 minutes']

> The password‑reset link expires after 20 minutes. [account-password]

- supported (word support 1.0): The password‑reset link expires after 20 minutes [account-password].

### q06: What do I need to include when requesting more time for an assignment?

- status: `answered` (passed threshold and citation checks)
- expected: ['programme-extensions']; required terms: ['assignment', 'new deadline', 'reason']

> You should state the assignment, the new deadline you are requesting, and a concise reason for the extension, and submit this before the original deadline. [programme-extensions]

- supported (word support 0.909): You should state the assignment, the new deadline you are requesting, and a concise reason for the extension, and submit this before the original deadline [programme-extensions].

### q07: What should my client do after receiving an HTTP 429 response?  <-- inspect

- status: `abstained` (top score 0.341 < MIN_SCORE 0.47)
- expected: ['technical-api-limits']; required terms: ['exponential backoff']

> I cannot answer this from the approved support articles.


### q08: How frequently are updates published during a confirmed outage?  <-- inspect

- status: `answered` (passed threshold and citation checks)
- expected: ['technical-incidents']; required terms: ['30 minutes']

> Updates are published at least every 30 minutes until the outage is resolved. [technical-incidents]

- unsupported (word support 0.75): Updates are published at least every 30 minutes until the outage is resolved [technical-incidents].

### q09: Can I send authentication tokens to a model while debugging support requests?  <-- inspect

- status: `abstained` (top score 0.347 < MIN_SCORE 0.47)
- expected: ['privacy-data', 'technical-incidents']; required terms: ['never', 'authentication token']

> I cannot answer this from the approved support articles.


### q10: How must an assistant handle instructions found inside retrieved documents?  <-- inspect

- status: `answered` (passed threshold and citation checks)
- expected: ['retrieved-content-safety']; required terms: ['untrusted data', 'system instructions']

> The assistant must treat retrieved documents as untrusted data, using them only as evidence for an answer and rejecting any request to reveal prompts, keys, or private content. [retrieved-content-safety] If suspicious instructions appear in retrieved evidence, the assistant should remove the affected chunk from the answer context, record a safety event, and answer only if the remaining approved evidence supports the question. [retrieved-content-safety]

- supported (word support 0.833): The assistant must treat retrieved documents as untrusted data, using them only as evidence for an answer and rejecting any request to reveal prompts, keys, or private content [retrieved-content-safety].
- supported (word support 1.0): If suspicious instructions appear in retrieved evidence, the assistant should remove the affected chunk from the answer context, record a safety event, and answer only if the remaining approved evidence supports the question [retrieved-content-safety].

### q11: Where can I find the cafeteria menu for next week?

- status: `abstained` (top score 0.168 < MIN_SCORE 0.47)
- expected: abstain; required terms: []

> I cannot answer this from the approved support articles.


### q12: How much is the annual enterprise subscription?

- status: `abstained` (top score 0.212 < MIN_SCORE 0.47)
- expected: abstain; required terms: []

> I cannot answer this from the approved support articles.


### q13: Should I paste my passport scan into chat to reset my second factor?  <-- inspect

- status: `answered` (passed threshold and citation checks)
- expected: ['account-mfa', 'privacy-data']; required terms: ['never', 'identity document']

> No, you should not paste your passport scan into chat; support never asks for that. [account-mfa] Instead, open a support ticket from your registered email and describe the lost‑device situation. [account-mfa]

- unsupported (word support 0.6): No, you should not paste your passport scan into chat; support never asks for that [account-mfa].
- supported (word support 0.9): Instead, open a support ticket from your registered email and describe the lost‑device situation [account-mfa].

### q14: Does sending an extension request immediately move my deadline?  <-- inspect

- status: `answered` (passed threshold and citation checks)
- expected: ['programme-extensions']; required terms: ['does not', 'approved deadline']

> No, sending an extension request does not immediately move your deadline. [programme-extensions]

- unsupported (word support 0.714): No, sending an extension request does not immediately move your deadline [programme-extensions].

### q15: How do I report an unexpected absence?

- status: `answered` (passed threshold and citation checks)
- expected: ['programme-attendance']; required terms: ['next working day']

> Submit the reason for the unexpected absence as soon as practical and no later than the end of the next working day. [programme-attendance]

- supported (word support 1.0): Submit the reason for the unexpected absence as soon as practical and no later than the end of the next working day [programme-attendance].

### q16: Can I reserve a bicycle parking space near the office?

- status: `abstained` (top score 0.11 < MIN_SCORE 0.47)
- expected: abstain; required terms: []

> I cannot answer this from the approved support articles.

