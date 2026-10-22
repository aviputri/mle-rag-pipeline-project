# RAG Pipeline Project: Evaluation Report

**Recommendation in one line:** the assistant is safe enough to continue as a
supervised pilot (no false answers, no fabricated citations on the final set),
but not yet reliable enough for unsupervised use: it refused 2 of 13 answerable
questions, and both evaluation sets are too small to support production claims.

## 1. System and configuration

| Component | Setting |
|---|---|
| Knowledge base | 10 Markdown articles in `data/knowledge_base/` (unchanged) |
| Chunking | one paragraph = one chunk; paragraphs < 150 chars merged with the next; paragraphs > 500 chars split at sentence boundaries with 1-sentence overlap → **21 chunks** |
| Chunk metadata | `chunk_id` (`<document_id>#<index>`), `document_id`, `title`, `section`, `chunk_index`, `text`, `source_path`, `doc_sha256` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dims, normalised, CPU, via `langchain_huggingface` |
| Embedded text | title + paragraph (plain paragraph kept in metadata for citing) |
| Vector store | Qdrant local mode via `langchain_qdrant`, collection `northstar_kb`, cosine distance, on disk at `storage/qdrant` (rebuildable, git-ignored) |
| Retrieval | top-k = **3** |
| Abstention | answer only if the best *safe* top-1 score ≥ **0.47**; all safe top-3 chunks go to the generator |
| Guardrails | regex input guard; retrieved chunks with instruction-like text are dropped and logged; context wrapped in `<source id="…">` tags declared as untrusted data; citations must be retrieved document IDs |
| Generation (baseline) | `extractive`: deterministic quote of the best-supported document's retrieved chunks (highest summed score) |
| Generation (live) | `llm:groq/openai/gpt-oss-20b`, temperature 0, LangChain `ChatPromptTemplate │ ChatOpenAI │ StrOutputParser`, prompt adapted from course notebook 04 (sha1 `f446011993`) |

Rebuild the index from the sources: `uv run python -m src.retrieval rebuild`.
Settings were frozen in `.env.example` (commit `a8913c0`) before the final run.

## 2. Threshold selection (calibration set only)

Selected on `data/evaluation/calibration.json` (6 questions, 3 should abstain),
before `questions.json` was run. Per-question signals with `k = 3`:

| id | expected | top-1 score | margin to #2 | rank of expected doc | rank of answer chunk |
|---|---|---|---|---|---|
| cal01 | billing-refunds | 0.550 | 0.017 | 2 | 3 |
| cal02 | account-mfa | 0.496 | 0.131 | 1 | 1 |
| cal03 | technical-api-limits | 0.708 | 0.316 | 1 | 1 |
| cal04 | abstain | 0.310 | 0.018 | – | – |
| cal05 | abstain | 0.441 | 0.049 | – | – |
| cal06 | abstain | 0.294 | 0.042 | – | – |

Rule A (top-1 score ≥ t) sweep: 0.40 → 1 false answer; **0.45 → 0 errors**;
0.50 → 1 false refusal. Supported top scores (0.496–0.708) and unsupported top
scores (0.294–0.441) were separable, so the threshold was set at the **midpoint
of the gap, 0.47**, rather than at 0.45, which sat only 0.009 above cal05.

Rejected alternatives:

- **Margin rule:** cal01 is supported but its margin is only 0.017, so any useful margin requirement refuses it.
- **Query-overlap rule:** tied with rule A on calibration and adds complexity, so the simpler rule was kept.
- **k = 5:** changed no decision or metric. k < 3 would lose cal01's answer chunk, which is at rank 3.
- **Per-chunk threshold** (as in notebook 04): would drop cal01's answer chunk (0.446). Hence the threshold applies to the top-1 score only.

## 3. Final evaluation (`questions.json`, 16 questions, run once per mode)

Both modes use the same retrieval and the same decisions; only the answer text differs.

| Metric | Extractive | LLM (gpt-oss-20b) |
|---|---|---|
| hit@3 | 1.000 | 1.000 |
| recall@3 | 1.000 | 1.000 |
| MRR | 0.923 | 0.923 |
| Decision accuracy | **0.875** (14/16) | **0.875** (14/16) |
| False answers (answered an unsupported question) | **0** | **0** |
| False refusals (refused a supported question) | 2 (q07, q09) | 2 (q07, q09) |
| Answer relevance | 0.846 | 0.654 |
| Faithfulness (automatic rubric) | 1.000 (11/11 literal) | 0.773 (8/11 faithful) |
| Faithfulness (after manual review) | 1.000 | 1.000 (11/11) |
| Citation precision | 1.000 | 1.000 |
| Fabricated or invalid citations | 0 | 0 |

### Metric rubrics

- **hit@3 / recall@3 / MRR** (13 supported questions, raw top-3):
  - **hit@3:** whether any expected document is retrieved.
  - **recall@3:** the share of expected documents retrieved (matters for the two-source questions q09 and q13).
  - **MRR:** the mean of 1 / rank of the first expected document.
- **Decision accuracy:** a question is correct if it is answered and supported, or not answered and unsupported. `abstained`, `blocked_input` and `citation_failed` all count as not answered.
- **Answer relevance:** the share of the question's `required_terms` found in the answer (case-insensitive). An abstention on a supported question scores 0.
- **Faithfulness:** each answer sentence is compared with the retrieved chunks of the document it cites, and labelled:
  - **literal** if it appears word for word;
  - **supported** if at least 80% of its content words (after simple suffix stripping) and every number appear in that evidence;
  - **unsupported** otherwise.

  The answer's score is the share of literal or supported sentences. Non-literal sentences are flagged for manual review, because word overlap cannot prove that a paraphrase means the same thing.
- **Citation precision:** the share of cited document IDs that are in `expected_sources`.

### Per-question results

| id | expected | top-3 retrieved (score) | decision (both modes) | expected doc rank | relevance E / L | LLM faithfulness auto → manual |
|---|---|---|---|---|---|---|
| q01 | billing-refunds | invoice-corrections#1 (0.634), billing-refunds#2 (0.505), billing-refunds#0 (0.470) | answered ✔ | 2 | 1.00 / 1.00 | 1.00 → ✔ |
| q02 | billing-refunds | billing-refunds#1 (0.756), #0 (0.534), #2 (0.522) | answered ✔ | 1 | 1.00 / 1.00 | 1.00 (literal) |
| q03 | invoice-corrections | invoice-corrections#0 (0.762), #1 (0.585), billing-refunds#2 (0.434) | answered ✔ | 1 | 1.00 / 0.50 | 1.00 → ✔ |
| q04 | account-mfa | account-mfa#0 (0.623), #1 (0.491), account-password#1 (0.283) | answered ✔ | 1 | 1.00 / 0.50 | 1.00 → ✔ |
| q05 | account-password | account-password#0 (0.821), #1 (0.692), account-mfa#0 (0.331) | answered ✔ | 1 | 1.00 / 1.00 | 1.00 → ✔ |
| q06 | programme-extensions | programme-extensions#0 (0.518), #1 (0.416), programme-attendance#0 (0.329) | answered ✔ | 1 | 1.00 / 1.00 | 1.00 → ✔ |
| q07 | technical-api-limits | technical-incidents#1 (0.341), technical-api-limits#0 (0.305), retrieved-content-safety#2 (0.302) | **abstained ✘** | 2 | 0 / 0 | – |
| q08 | technical-incidents | technical-incidents#0 (0.680), #1 (0.424), programme-attendance#0 (0.248) | answered ✔ | 1 | 1.00 / 1.00 | 0.00 → ✔ |
| q09 | privacy-data, technical-incidents | technical-incidents#1 (0.347), privacy-data#0 (0.328), account-mfa#1 (0.303) | **abstained ✘** | 1 | 0 / 0 | – |
| q10 | retrieved-content-safety | retrieved-content-safety#0 (0.620), #2 (0.487), privacy-data#0 (0.414) | answered ✔ | 1 | 1.00 / 0.50 | 1.00 → ✔ |
| q11 | abstain | programme-attendance#0 (0.168), … | abstained ✔ | – | – | – |
| q12 | abstain | technical-api-limits#0 (0.212), … | abstained ✔ | – | – | – |
| q13 | account-mfa, privacy-data | account-mfa#1 (0.557), #0 (0.391), privacy-data#1 (0.320) | answered ✔ | 1 | 1.00 / 0.50 | 0.50 → ✔ (partial: cites one of two sources) |
| q14 | programme-extensions | programme-extensions#1 (0.771), #0 (0.623), technical-incidents#0 (0.287) | answered ✔ | 1 | 1.00 / 0.50 | 0.00 → ✔ |
| q15 | programme-attendance | programme-attendance#0 (0.590), #1 (0.452), privacy-data#1 (0.289) | answered ✔ | 1 | 1.00 / 1.00 | 1.00 → ✔ |
| q16 | abstain | programme-attendance#1 (0.110), … | abstained ✔ | – | – | – |

Full answers: `reports/evaluation/results_extractive.md` and `results_llm.md`.

## 4. Failure and fragility analysis

### F1: q07 refused, with a ranking failure behind it

*"What should my client do after receiving an HTTP 429 response?"* The answer
("retry using exponential backoff") is in `technical-api-limits#0`. That chunk
was ranked **second (0.305)**, behind `technical-incidents#1` (0.341), which is
about slow requests and API keys in tickets. The top-1 score of 0.341 is below
0.47, so both modes abstained.

The query and the answer share the exact token `429`, but MiniLM gives
short technical tokens little weight. A keyword (BM25) component would match it
directly. The refusal was safe, because nothing wrong was said, but it pushes a
routine question to escalation.

### F2: q09 refused, with evidence split across two documents

*"Can I send authentication tokens to a model while debugging support
requests?"* Both expected documents were retrieved: `technical-incidents#1`
(0.347) and `privacy-data#0` (0.328, which contains the answer: *never include …
authentication tokens … in a language-model prompt*). Neither reached 0.47.

Questions that combine two topics match each document only partly, so their
best single score stays low. The threshold uses the best single chunk score, so it
penalises exactly these multi-source questions.

### F3: q01 / cal01, a fragile success caused by a cross-reference chunk

The top hit for the duplicate-payment questions is `invoice-corrections#1`
(0.634 on q01, 0.550 on cal01). It is a pointer sentence that shares the
question's wording (*"Duplicate charges follow the refund procedure…"*). The
paragraph that actually answers is `billing-refunds#0` at **rank 3 (0.470 / 0.446)**.

The question is answered correctly only because of two design choices:

- all top-3 chunks are passed to the generator, not only those above the threshold;
- the extractive mode picks the document with the highest *summed* score.

With k = 2, with a per-chunk threshold, or with "quote the top chunk" logic, this
case fails.

### F4: Generation-side measurement problems (LLM mode)

**A scorer bug, fixed after the first scoring.** gpt-oss-20b outputs typographic
characters (`30 minutes`, `lost‑device`), so plain matching scored q08's
correct answer as relevance 0. The fix normalises these characters to ASCII before
matching. The **saved answers were re-scored without regeneration**
(`--rescore`), and the pipeline itself was not changed. The original scoring is
kept in git history (commit `9e48504`).

Effect of the fix:

- answer relevance: 0.577 → 0.654;
- answers needing review: 11 → 10.

**Lexical false negatives.** The rubric marked three LLM sentences unsupported
(q08, q13, q14). On manual review, all three are faithful paraphrases:

- **q08:** "published … until resolved" vs. "posts … until resolution".
- **q14:** "does not immediately move your deadline" vs. "does not itself change the submission deadline".
- **q13:** "support never asks for that" vs. "Support never asks for … passport scan … in chat".

**Lower LLM relevance (0.654 vs 0.846) is partly how the rubric works.** Extractive
answers quote whole chunks, so required terms appear even when they go beyond the
question. The LLM answers concisely and paraphrases terms ("after verification"
instead of "identity verification", "passport scan" instead of "identity document").

**Real gap:** q13 cites only `account-mfa`. The privacy rule in `privacy-data` was
retrieved but not used, so the answer is correct but partial for a two-source
question. The extractive mode has the same limit by design: it cites a single
document.

### Label notes

The labels were not edited. Two are debatable:

- **q03** requires "invoice number", although the question only asks whether
  the name can be corrected.
- **q13** expects `privacy-data`, whose rule covers language-model prompts rather
  than chat; `account-mfa` alone answers the chat question.

Both lower LLM relevance, but the scores are reported as labelled.

## 5. Threshold in hindsight

On the final set, supported top-1 scores ranged from **0.341 to 0.821** and
unsupported ones from **0.110 to 0.212**. A threshold around 0.30 would have
decided all 16 correctly. It was **not** changed after seeing these results.

The calibration set contained a much harder near-miss (cal05, 0.441, about
annual-subscription refunds), while the final set's unsupported questions were
unrelated topics (cafeteria, bicycle parking, enterprise pricing). The calibration
set was therefore not representative of the final distribution in either
direction:

- it had no short technical paraphrases like q07;
- it had no two-source questions like q09.

## 6. Prioritised recommendations

1. **Recalibrate abstention on a larger, representative set** (for example 50+
   questions, including paraphrases, short technical queries, two-source
   questions and near-misses). Evaluate a lower score threshold combined with the
   LLM's "cannot answer" layer, measuring false answers on near-misses
   explicitly. *Evidence:* F1 and F2 were refused at 0.341 and 0.347; unsupported
   questions scored at most 0.212; the single calibration near-miss (0.441) set
   the threshold.
2. **Add hybrid retrieval (BM25 + dense, for example Qdrant sparse vectors through
   `langchain_qdrant` hybrid mode) or a cross-encoder reranker** over the top-k.
   *Evidence:* exact-token misses (q07, `429`) and cross-reference chunks
   outranking the policy (q01 and cal01, F3). MRR is 0.923 only because the right
   document was at rank 2 twice.
3. **Make answer-quality checking independent and multi-source aware.** Use a
   documented human review protocol, or a validated LLM judge, for paraphrased
   answers. Allow and require citations for every needed source.
   *Evidence:* the lexical rubric produced three false "unsupported" labels and
   missed a Unicode issue (F4); q13 cited one of two sources.

## 7. Submission questions

**1. Which retrieval failures matter most for support, and why?**

The most harmful failure would be confidently answering from a policy that
doesn't apply: a refund rule applied to a cancellation (cal05), or an
invoice-correction pointer treated as the refund policy (F3). That leads
colleagues to tell customers something wrong. The system had **0 false answers**
on both sets.

Next come missed or misranked evidence for routine questions (q07) and
multi-source questions whose scores are split (q09). These turn into refusals,
which are safe but add escalations and reduce trust. Exact-token mismatches
(error codes, product terms) are the most common cause seen here.

**2. How did you select your no-answer threshold?**

Only on `calibration.json`, before running `questions.json`. I measured
top-1 scores and swept score, margin and overlap rules. Supported and unsupported
questions were separable by top-1 score (0.496 vs 0.441), and 0.47 is the
midpoint of that gap. The margin and overlap rules were rejected (section 2). The
settings were frozen in `.env.example` and committed before the final run.

**3. What evidence suggests an answer is faithful to retrieved context?**

- Every answer passed the citation check: each cited ID was among the retrieved
  documents, and there were no uncited answers.
- Every extractive sentence appears **literally** in the cited chunks (11/11).
- LLM sentences were checked per sentence against the cited chunks (word and
  number support), then manually reviewed: 11/11 faithful, with q13 partial.

Citation correctness alone is not proof. A correctly cited sentence can still
misstate its source, which is why the per-sentence check and the manual review
exist. The automatic rubric is a lexical proxy, not a semantic judgement.

**4. What would you change before live support colleagues use it?**

- The three recommendations above.
- A larger, versioned evaluation set re-run in CI on every change (prompt, model, chunking).
- Routing every abstention to a clear escalation path instead of a dead end.
- Logging of decisions, scores, citations and safety events (without personal data).
- A provider fallback, with extractive mode as the default when the LLM is unavailable.
- Redaction of personal data before any text is sent to the LLM provider, as required by `privacy-data`.
- A supervised pilot in which colleagues rate answers before unsupervised use.

## 8. Reproducing the results

```bash
uv sync
cp .env.example .env                                   # add GROQ_API_KEY for LLM mode
uv run python -m src.retrieval rebuild
uv run pytest -q                                       # 76 offline tests
uv run python -m src.calibration                       # calibration evidence
uv run python -m src.evaluation --mode extractive      # deterministic baseline
uv run python -m src.evaluation --mode llm             # live generation (may vary slightly)
```

Evidence files:

- `reports/calibration/`
- `reports/evaluation/results_*.{md,json}`
- `reports/demos/` (cited answer, abstention, blocked input, retrieved-injection drop)

Both sets are small teaching fixtures (6 and 16 questions). The numbers describe
behaviour on these cases, not general production reliability.
