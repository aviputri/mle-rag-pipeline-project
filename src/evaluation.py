"""Final evaluation: run every question through the SAME pipeline as ``ask``.

    uv run python -m src.evaluation --mode extractive   # deterministic, no key
    uv run python -m src.evaluation --mode llm          # Groq (needs GROQ_API_KEY)

Run this once per mode, AFTER the settings are frozen (TOP_K, MIN_SCORE,
prompt). Do not change settings based on these results; report them instead.

Metrics and rubrics
-------------------
Retrieval (supported questions only, on the raw top-k before any filtering):
* ``hit@k``     1 if any expected document is in the top-k, else 0.
* ``recall@k``  share of the expected documents found in the top-k
                (differs from hit@k for multi-source questions).
* ``MRR``       mean of 1 / rank of the first expected document (miss = 0).

Decision (all questions):
* correct = (status is "answered") XOR (should_abstain).
  "abstained", "blocked_input", "citation_failed" all count as not answering.
* ``decision accuracy`` = correct / all.

Answer relevance (supported questions; deterministic rubric):
* share of the question's ``required_terms`` that appear in the answer text
  (case-insensitive). An abstention on a supported question scores 0.

Faithfulness (answered questions; deterministic rubric, per sentence):
* Support text = the retrieved chunks given to the generator that belong to
  the documents cited in that sentence (all context chunks if uncited).
* ``literal``     the sentence appears verbatim in the support text.
* ``supported``   not verbatim, but >= 80% of its content words occur in the
                  support text (after stripping simple suffixes, so
                  requester ~ request, refunds ~ refund) AND every number in
                  it occurs there too.
* ``unsupported`` anything else.
* Answer faithfulness = share of literal/supported sentences; an answer is
  faithful if that share is 1. Non-literal sentences are a lexical proxy
  only and are flagged ``needs_review`` for a human check: word overlap
  cannot prove that a paraphrase means the same thing.

Citation precision (answered supported questions):
* share of cited document IDs that are in ``expected_sources``.
"""

# Allows "str | None" style type hints to be used everywhere without runtime cost.
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from src.calibration import content_words
from src.generation import SYSTEM_PROMPT, cited_ids
from src.ingestion import _SENTENCE_RE, build_chunks
from src.pipeline import RAGAnswer, RAGPipeline

# ---------------------------------------------------------------- settings

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "evaluation" / "questions.json"
DEFAULT_OUT_DIR = ROOT / "reports" / "evaluation"

# A paraphrased sentence counts as supported if at least this share of its
# content words appears in the cited evidence.
SUPPORT_THRESHOLD = 0.8

_CITATION_RE = re.compile(r"\[[^\[\]]+\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


# ------------------------------------------------------------ rubrics


def normalise(text: str) -> str:
    """Lower-case, single spaces, no citations, no backticks: for literal matching."""
    text = _CITATION_RE.sub(" ", text).replace("`", "")
    return " ".join(text.lower().split())


def split_sentences(answer: str) -> list[str]:
    """Answer -> sentences, each keeping its own citation(s).

    Citations follow the full stop ("... day. [doc] Next ..."), so we first
    move every citation in front of the punctuation it follows.
    """
    moved = re.sub(r"([.!?])\s*((?:\[[^\[\]]+\]\s*)*\[[^\[\]]+\])\s*", r" \2\1 ", answer)
    return [s.strip() for s in _SENTENCE_RE.split(moved.strip()) if s.strip()]


# Suffixes removed before comparing words, longest first.
_SUFFIXES = ("ations", "ation", "ions", "ion", "ing", "ers", "er", "ed", "es", "s", "ly")


def stem(word: str) -> str:
    """Very small stemmer: requester -> request, refunds -> refund.

    Stricter than the 5-letter prefix match used for retrieval signals, which
    wrongly equated e.g. "correction" and "correspond".
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def judge_sentence(sentence: str, support: str) -> dict:
    """Apply the per-sentence faithfulness rubric."""
    clean = normalise(sentence).rstrip(".!? ")
    support_norm = normalise(support)
    if clean and clean in support_norm:
        return {"sentence": sentence, "label": "literal", "word_support": 1.0}

    words = {stem(w) for w in content_words(clean)}
    support_words = {stem(w) for w in content_words(support_norm)}
    share = (len(words & support_words) / len(words)) if words else 0.0
    # Every number (days, limits, minutes) must come from the evidence.
    numbers_ok = all(n in support_norm for n in _NUMBER_RE.findall(clean))
    label = "supported" if share >= SUPPORT_THRESHOLD and numbers_ok else "unsupported"
    return {"sentence": sentence, "label": label, "word_support": round(share, 3),
            "numbers_ok": numbers_ok}


def faithfulness(result: RAGAnswer, chunk_text: dict[str, str]) -> dict:
    """Judge every sentence of an answered result against its cited evidence."""
    # Context chunks actually given to the generator, grouped by document.
    context = {cid: chunk_text[cid] for cid in result.context_ids}
    judged = []
    for sentence in split_sentences(result.answer):
        cited = set(cited_ids(sentence))
        # Evidence = context chunks of the cited documents (all if uncited).
        support = " ".join(
            text for cid, text in context.items()
            if not cited or cid.split("#")[0] in cited
        )
        verdict = judge_sentence(sentence, support)
        verdict["cited"] = sorted(cited)
        judged.append(verdict)

    ok = [j for j in judged if j["label"] in ("literal", "supported")]
    score = len(ok) / len(judged) if judged else 0.0
    return {
        "score": round(score, 3),
        "faithful": score == 1.0,
        "needs_review": any(j["label"] != "literal" for j in judged),
        "sentences": judged,
    }


def term_coverage(answer: str, terms: list[str]) -> float | None:
    """Answer relevance rubric: share of required terms present in the answer."""
    if not terms:
        return None
    lowered = answer.lower()
    return sum(t.lower() in lowered for t in terms) / len(terms)


def retrieval_metrics(retrieved_docs: list[str], expected: list[str]) -> dict:
    """hit@k, recall@k and reciprocal rank for one supported question."""
    ranks = [i for i, doc in enumerate(retrieved_docs, start=1) if doc in expected]
    found = set(retrieved_docs) & set(expected)
    return {
        "hit": int(bool(ranks)),
        "recall": len(found) / len(expected),
        "reciprocal_rank": 1 / ranks[0] if ranks else 0.0,
        "first_rank": ranks[0] if ranks else None,
    }


# --------------------------------------------------------------- one case


def evaluate_case(case: dict, result: RAGAnswer, chunk_text: dict[str, str]) -> dict:
    """All metrics for one question, plus the raw answer record."""
    answered = result.status == "answered"
    should_abstain = case["should_abstain"]
    row = {
        "id": case["id"],
        "question": case["question"],
        "should_abstain": should_abstain,
        "expected_sources": case["expected_sources"],
        "required_terms": case["required_terms"],
        "status": result.status,
        "reason": result.reason,
        "decision_correct": answered != should_abstain,
        "top_score": result.top_score,
        "retrieved": result.retrieved,
        "answer": result.answer,
        # What the generator wrote before the checks (differs for citation_failed).
        "raw_answer": getattr(result, "raw_answer", ""),
        "citations": result.citations,
        "invalid_citations": result.invalid_citations,
        "safety_events": result.safety_events,
    }

    if not should_abstain:
        docs = [r["document_id"] for r in result.retrieved]
        row["retrieval"] = retrieval_metrics(docs, case["expected_sources"])
        # Abstaining on a supported question = no relevant answer = 0.
        row["relevance"] = term_coverage(result.answer, case["required_terms"]) if answered else 0.0
        if answered and result.citations:
            good = [c for c in result.citations if c in case["expected_sources"]]
            row["citation_precision"] = len(good) / len(result.citations)

    if answered:
        row["faithfulness"] = faithfulness(result, chunk_text)
    return row


def summarise(rows: list[dict]) -> dict:
    """Aggregate metrics over all rows."""
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    supported = [r for r in rows if not r["should_abstain"]]
    unsupported = [r for r in rows if r["should_abstain"]]
    answered = [r for r in rows if r["status"] == "answered"]
    return {
        "cases": len(rows),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "hit@k": mean(r["retrieval"]["hit"] for r in supported),
        "recall@k": mean(r["retrieval"]["recall"] for r in supported),
        "mrr": mean(r["retrieval"]["reciprocal_rank"] for r in supported),
        "decision_accuracy": mean(int(r["decision_correct"]) for r in rows),
        "answered_supported": sum(r["status"] == "answered" for r in supported),
        "abstained_unsupported": sum(r["status"] != "answered" for r in unsupported),
        "false_answers": sum(r["status"] == "answered" for r in unsupported),
        "false_refusals": sum(r["status"] != "answered" for r in supported),
        "answer_relevance": mean(r.get("relevance") for r in supported),
        "faithfulness": mean(r["faithfulness"]["score"] for r in answered),
        "faithful_answers": sum(r["faithfulness"]["faithful"] for r in answered),
        "answers_needing_review": sum(r["faithfulness"]["needs_review"] for r in answered),
        "citation_precision": mean(r.get("citation_precision") for r in supported),
        "status_counts": {s: sum(r["status"] == s for r in rows)
                          for s in ("answered", "abstained", "blocked_input",
                                    "citation_failed", "error")},
    }


# ----------------------------------------------------------------- output


def _f(value, digits: int = 2) -> str:
    # Table formatting: numbers with fixed decimals, "-" for missing.
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(rows: list[dict], summary: dict, meta: dict) -> str:
    """Readable results page: settings, metrics, per-question table, details."""
    lines = [
        f"# Evaluation Results ({meta['mode']})",
        "",
        f"- Questions: `{meta['questions']}` ({summary['cases']} cases: "
        f"{summary['supported']} supported, {summary['unsupported']} should abstain)",
        f"- Generation mode: `{meta['mode']}`",
        f"- Embedding: `{meta['embedding_model']}` @ `{meta['embedding_revision']}`",
        f"- TOP_K = {meta['top_k']}, MIN_SCORE = {meta['min_score']} (frozen after calibration)",
        f"- System prompt sha1: `{meta['prompt_sha1']}`, git commit: `{meta['git_commit']}`",
        f"- Run at {meta['run_at']} ({meta['seconds']} s)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ("hit@k", "recall@k", "mrr", "decision_accuracy", "false_answers",
                "false_refusals", "answer_relevance", "faithfulness",
                "faithful_answers", "answers_needing_review", "citation_precision"):
        lines.append(f"| {key} | {_f(summary[key], 3)} |")
    lines += ["", f"Status counts: {summary['status_counts']}", "",
              "## Per-question results", "",
              "| id | expected | top-3 retrieved (score) | top | status | decision | first rank | relevance | faithfulness | cited |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        retrieved = ", ".join(f"{x['chunk_id']} ({x['score']:.3f})" for x in r["retrieved"]) or "-"
        faith = r.get("faithfulness")
        faith_cell = "-" if not faith else (
            f"{faith['score']:.2f}" + (" (review)" if faith["needs_review"] else ""))
        lines.append(
            f"| {r['id']} | {', '.join(r['expected_sources']) or 'ABSTAIN'} | {retrieved} "
            f"| {_f(r['top_score'], 3)} | {r['status']} | {'ok' if r['decision_correct'] else 'WRONG'} "
            f"| {_f(r.get('retrieval', {}).get('first_rank'))} | {_f(r.get('relevance'))} "
            f"| {faith_cell} | {', '.join(r['citations']) or '-'} |"
        )
    lines += ["", "## Answers and failure details", ""]
    for r in rows:
        flag = "" if r["decision_correct"] and (r.get("relevance") in (None, 1.0)) else "  <-- inspect"
        lines += [f"### {r['id']}: {r['question']}{flag}", "",
                  f"- status: `{r['status']}` ({r['reason']})",
                  f"- expected: {r['expected_sources'] or 'abstain'}; required terms: {r['required_terms']}"]
        if r["safety_events"]:
            lines.append(f"- safety: {r['safety_events']}")
        if r["invalid_citations"]:
            lines.append(f"- invalid citations: {r['invalid_citations']}")
        lines += ["", f"> {r['answer']}", ""]
        if r["raw_answer"] and r["raw_answer"] != r["answer"]:
            lines += [f"Generator output before checks:", "", f"> {r['raw_answer']}", ""]
        for s in r.get("faithfulness", {}).get("sentences", []):
            if s["label"] != "literal":
                lines.append(f"- {s['label']} (word support {s['word_support']}): {s['sentence']}")
        lines.append("")
    return "\n".join(lines)


def git_commit() -> str:
    """Current commit hash, so results can be traced to exact code."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if out.returncode != 0:
            return "unknown"
        return out.stdout.strip() + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cases = json.loads(args.questions.read_text())
    # chunk_id -> text, for judging answers against exactly what was retrieved.
    chunk_text = {c.chunk_id: c.text for c in build_chunks()}
    # The same pipeline class the "ask" command uses.
    pipeline = RAGPipeline(mode=args.mode)

    start = time.time()
    rows = []
    for case in cases:
        try:
            result = pipeline.answer(case["question"])
        except Exception as exc:  # e.g. provider timeout: record it, keep going
            # model_construct skips validation, so the extra "error" status is allowed.
            result = RAGAnswer.model_construct(
                question=case["question"], status="error", answer="",
                reason=f"error: {exc!r}", citations=[], invalid_citations=[],
                retrieved=[], context_ids=[], top_score=None, safety_events=[],
                mode=pipeline.generator.mode, raw_answer="")
        rows.append(evaluate_case(case, result, chunk_text))
        print(f"{case['id']}: {rows[-1]['status']:<15} "
              f"{'ok   ' if rows[-1]['decision_correct'] else 'WRONG'} {case['question']}")

    summary = summarise(rows)
    meta = {
        "questions": str(args.questions.relative_to(ROOT)) if args.questions.is_relative_to(ROOT) else str(args.questions),
        "mode": pipeline.generator.mode,
        "embedding_model": pipeline.config.embedding_model,
        "embedding_revision": pipeline.config.embedding_revision,
        "top_k": pipeline.config.top_k,
        "min_score": pipeline.min_score,
        "prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:10],
        "git_commit": git_commit(),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seconds": round(time.time() - start, 1),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "extractive" if args.mode == "extractive" else "llm"
    (args.out_dir / f"results_{name}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2))
    (args.out_dir / f"results_{name}.md").write_text(markdown(rows, summary, meta))

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nSaved reports/evaluation/results_{name}.md and .json")


if __name__ == "__main__":
    main()
