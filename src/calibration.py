"""
Calibrate retrieval and abstention settings on ``calibration.json``.

For every calibration question
--> record retrieval returns & evidence signals
--> then sweep candidate abstention rules and calculate errors
--> output: the evidence for the "chosen threshold" section of the report

Signals per question
--------------------
* ``top_score``      cosine similarity of the rank-1 chunk.
* ``margin``         top_score minus the rank-2 score.
* ``doc_rank``       rank of the first chunk from an expected document.
* ``evidence_rank``  rank of the first chunk that contains ALL required terms
                     (the chunk that can actually answer the question).
* ``term_coverage``  share of required terms found anywhere in the top-k text.
* ``query_overlap``  share of the question's content words found in the top-k text.

Candidate rules (answer only if ...)
------------------------------------
A. ``top_score >= t``
B. ``top_score >= t`` and ``margin >= m``
C. ``top_score >= t`` and ``query_overlap >= o``

Errors:
false answer: answering a ``should_abstain`` question (a confident wrong policy)
false refusal: abstaining on a supported one (escalation)
--> In support, a false answer is worse than a false refusal

Only use ``calibration.json`` here.
Freeze the chosen settings in ``.env``
BEFORE running the final ``questions.json`` evaluation.

``uv run python -m src.calibration``
``uv run python -m src.calibration -k 5``
"""

# Allows "str | None" style type hints to be used everywhere without runtime cost.
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_qdrant import QdrantVectorStore

from src.retrieval import (
    COLLECTION_NAME,
    RetrievalConfig,
    RetrievedChunk,
    build_vector_store,
    retrieve,
)

# ---------------------------------------------------------------- settings

# <repo> folder, resolved from this file so the script works from any folder.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "evaluation" / "calibration.json"
DEFAULT_OUT_DIR = ROOT / "reports" / "calibration"

# Grids of candidate values for the sweeps (score steps of 0.05).
SCORE_GRID = [round(0.20 + 0.05 * i, 2) for i in range(11)]       # 0.20 .. 0.70
MARGIN_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
OVERLAP_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Common English words ignored when measuring query_overlap; they appear in
# almost every text and say nothing about the subject.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can", "do",
    "does", "for", "from", "how", "i", "if", "in", "is", "it", "my", "of", "on",
    "or", "our", "should", "the", "there", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your", "must",
    "any", "have", "has", "me", "we",
}
# A "word" = a run of letters/digits (apostrophes and hyphens allowed inside).
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")


# ------------------------------------------------------------- data shapes


# Everything measured for one calibration question.
@dataclass
class CalibrationRow:
    id: str
    question: str
    should_abstain: bool
    expected_sources: list[str]
    required_terms: list[str]
    retrieved: list[dict]          # [{rank, score, document_id, chunk_id}, ...]
    top_score: float
    margin: float | None           # None when k == 1 (no rank-2 score)
    doc_rank: int | None           # None = expected doc not in top-k (or abstain case)
    evidence_rank: int | None      # None = no single chunk has all required terms
    term_coverage: float | None    # None for abstain cases (no required terms)
    query_overlap: float


# ----------------------------------------------------------------- signals


def content_words(text: str) -> set[str]:
    """Lower-cased words minus stopwords and very short tokens."""
    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in STOPWORDS and len(w) > 2
    }


def word_matches(word: str, vocabulary: set[str]) -> bool:
    """Exact match, or a shared 5-letter prefix (refund ~ refunds, extend ~ extension)."""
    # A crude stand-in for stemming, so simple plural/verb forms still count.
    return word in vocabulary or (
        len(word) >= 5 and any(v[:5] == word[:5] for v in vocabulary if len(v) >= 5)
    )


def query_overlap(question: str, hits: list[RetrievedChunk]) -> float:
    """Share of the question's content words that appear in the retrieved text."""
    query = content_words(question)
    # Pool all words of all retrieved chunks (title included).
    vocabulary = set().union(*(content_words(f"{h.title} {h.text}") for h in hits))
    if not query:
        return 0.0
    return sum(word_matches(w, vocabulary) for w in query) / len(query)


def contains_all(text: str, terms: list[str]) -> bool:
    # Case-insensitive literal check, e.g. "invoice number" in the chunk text.
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def measure(case: dict, hits: list[RetrievedChunk]) -> CalibrationRow:
    """Compute all signals for one question from its retrieved chunks."""
    expected = set(case["expected_sources"])
    terms = case["required_terms"]

    # First rank whose document is one of the expected sources (doc level).
    doc_rank = next((h.rank for h in hits if h.document_id in expected), None)
    # First rank whose chunk alone contains every required term.
    evidence_rank = next(
        (h.rank for h in hits if terms and contains_all(h.text, terms)), None
    )
    # How many required terms are present somewhere in the whole top-k context.
    context = " ".join(h.text for h in hits).lower()
    coverage = (
        sum(t.lower() in context for t in terms) / len(terms) if terms else None
    )

    return CalibrationRow(
        id=case["id"],
        question=case["question"],
        should_abstain=case["should_abstain"],
        expected_sources=case["expected_sources"],
        required_terms=terms,
        retrieved=[
            {"rank": h.rank, "score": round(h.score, 4),
             "document_id": h.document_id, "chunk_id": h.chunk_id}
            for h in hits
        ],
        top_score=hits[0].score,
        margin=hits[0].score - hits[1].score if len(hits) > 1 else None,
        doc_rank=doc_rank,
        evidence_rank=evidence_rank,
        term_coverage=coverage,
        query_overlap=query_overlap(case["question"], hits),
    )


def run_calibration(
    cases: list[dict], store: QdrantVectorStore, k: int
) -> list[CalibrationRow]:
    """Retrieve the top-k chunks for every case and measure it."""
    return [measure(case, retrieve(store, case["question"], k=k)) for case in cases]


# ------------------------------------------------------------ rule sweeps


def answers(row: CalibrationRow, t: float, m: float = 0.0, o: float = 0.0) -> bool:
    """Decision of the combined rule: True = answer, False = abstain."""
    # margin is None when k == 1; treat that as "no margin requirement".
    margin_ok = row.margin is None or row.margin >= m
    return row.top_score >= t and margin_ok and row.query_overlap >= o


def count_errors(rows: list[CalibrationRow], t: float, m: float = 0.0, o: float = 0.0) -> dict:
    """False answers, false refusals and decision accuracy for one setting."""
    false_answers = sum(r.should_abstain and answers(r, t, m, o) for r in rows)
    false_refusals = sum(not r.should_abstain and not answers(r, t, m, o) for r in rows)
    return {
        "min_score": t, "min_margin": m, "min_overlap": o,
        "false_answers": false_answers,
        "false_refusals": false_refusals,
        "decision_accuracy": round(1 - (false_answers + false_refusals) / len(rows), 3),
    }


def sweep(rows: list[CalibrationRow]) -> dict[str, list[dict]]:
    """Evaluate rules A, B and C over their grids."""
    return {
        # Rule A: score threshold only.
        "A_score": [count_errors(rows, t) for t in SCORE_GRID],
        # Rule B: score threshold + minimum margin over rank 2.
        "B_score_margin": [
            count_errors(rows, t, m=m) for t in SCORE_GRID for m in MARGIN_GRID
        ],
        # Rule C: score threshold + minimum query/evidence word overlap.
        "C_score_overlap": [
            count_errors(rows, t, o=o) for t in SCORE_GRID for o in OVERLAP_GRID
        ],
    }


def best(results: list[dict]) -> list[dict]:
    """Settings with the fewest errors; ties broken by fewer false answers."""
    # Fewest total errors over the whole grid ...
    lowest = min(r["false_answers"] + r["false_refusals"] for r in results)
    # ... keep every setting that reaches it ...
    winners = [r for r in results if r["false_answers"] + r["false_refusals"] == lowest]
    # ... and list the safest first (fewer false answers, then lower threshold).
    return sorted(winners, key=lambda r: (r["false_answers"], r["min_score"]))


def score_gap(rows: list[CalibrationRow]) -> dict:
    """Do supported and unsupported questions separate on top_score alone?"""
    supported = [r.top_score for r in rows if not r.should_abstain]
    unsupported = [r.top_score for r in rows if r.should_abstain]
    lowest_supported = min(supported) if supported else None
    highest_unsupported = max(unsupported) if unsupported else None
    separable = (
        lowest_supported is not None and highest_unsupported is not None
        and lowest_supported > highest_unsupported
    )
    return {
        "lowest_supported_top_score": lowest_supported,
        "highest_unsupported_top_score": highest_unsupported,
        "separable_by_score": separable,
        # If separable, the midpoint leaves equal room on both sides.
        "midpoint": round((lowest_supported + highest_unsupported) / 2, 3) if separable else None,
    }


# ---------------------------------------------------------------- output


def _fmt(value, digits: int = 3) -> str:
    # Pretty-print numbers with a fixed number of decimals; "-" for missing
    # values (None), e.g. no expected doc in the top-k.
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def rows_table(rows: list[CalibrationRow]) -> str:
    """Markdown table with one line per question."""
    lines = [
        "| id | expected | retrieved (chunk: score) | top | margin | doc rank | evidence rank | term cov. | query overlap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        # e.g. "invoice-corrections#1: 0.550, billing-refunds#2: 0.533, ..."
        retrieved = ", ".join(f"{h['chunk_id']}: {h['score']:.3f}" for h in r.retrieved)
        expected = ", ".join(r.expected_sources) or "ABSTAIN"
        lines.append(
            f"| {r.id} | {expected} | {retrieved} | {_fmt(r.top_score)} | {_fmt(r.margin)} "
            f"| {_fmt(r.doc_rank)} | {_fmt(r.evidence_rank)} | {_fmt(r.term_coverage, 2)} "
            f"| {_fmt(r.query_overlap, 2)} |"
        )
    return "\n".join(lines)


def sweep_table(results: list[dict], columns: list[str]) -> str:
    """Markdown table for a list of rule settings."""
    # The rule's own settings first, then its error counts.
    head = columns + ["false_answers", "false_refusals", "decision_accuracy"]
    # Header row + the "|---|---|" separator Markdown needs for a table.
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        lines.append("| " + " | ".join(_fmt(r[c], 2) for c in head) + " |")
    return "\n".join(lines)


def retrieval_summary(rows: list[CalibrationRow], k: int) -> dict:
    """recall@k and MRR (document level) plus evidence recall, supported cases only."""
    # Abstain cases have no expected source, so they are left out here.
    supported = [r for r in rows if not r.should_abstain]
    # "or 1" avoids dividing by zero if a question set had no supported cases.
    n = len(supported) or 1
    return {
        # Share of supported questions with an expected doc anywhere in the top k.
        f"recall@{k}": round(sum(r.doc_rank is not None for r in supported) / n, 3),
        # Mean of 1/rank of the first expected doc (a miss counts as 0).
        "mrr": round(sum(1 / r.doc_rank for r in supported if r.doc_rank) / n, 3),
        # Stricter: a single retrieved chunk contains all required terms.
        f"evidence_recall@{k}": round(
            sum(r.evidence_rank is not None for r in supported) / n, 3
        ),
    }


def write_report(rows, sweeps, gap, summary, config, k, questions_path, out_dir) -> Path:
    """Save JSON (raw data) and Markdown (readable) calibration results."""
    # Create reports/calibration/ if it does not exist yet.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Everything needed to reproduce and audit the choice, as machine-readable JSON.
    payload = {
        "questions": str(questions_path.relative_to(ROOT)) if questions_path.is_relative_to(ROOT) else str(questions_path),
        "config": {**asdict(config), "k": k, "collection": COLLECTION_NAME},
        "retrieval": summary,
        "score_gap": gap,
        "rows": [asdict(r) for r in rows],
        "sweeps": sweeps,
        "best": {name: best(results) for name, results in sweeps.items()},
    }
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2))

    # The same results as a readable Markdown page (paste into the report).
    md = [
        "# Calibration Results",
        "",
        f"- Questions: `{payload['questions']}` ({len(rows)} cases)",
        f"- Embedding: `{config.embedding_model}` @ `{config.embedding_revision}`",
        f"- k = {k}, collection `{COLLECTION_NAME}`",
        f"- Retrieval (supported cases): {', '.join(f'{key} = {v}' for key, v in summary.items())}",
        "",
        "## Per-question signals",
        "",
        rows_table(rows),
        "",
        "## Score separation",
        "",
        f"- Lowest supported top score: {_fmt(gap['lowest_supported_top_score'])}",
        f"- Highest unsupported top score: {_fmt(gap['highest_unsupported_top_score'])}",
        f"- Separable by score alone: {gap['separable_by_score']}"
        + (f" (midpoint {gap['midpoint']})" if gap["midpoint"] else ""),
        "",
        "## Rule A: score threshold",
        "",
        sweep_table(sweeps["A_score"], ["min_score"]),
        "",
        "## Best settings per rule",
        "",
    ]
    for name, columns in [
        ("A_score", ["min_score"]),
        ("B_score_margin", ["min_score", "min_margin"]),
        ("C_score_overlap", ["min_score", "min_overlap"]),
    ]:
        md += [f"### {name}", "", sweep_table(best(sweeps[name]), columns), ""]
    md += [
        "> With only 6 calibration cases, many settings tie. Prefer the simplest",
        "> rule, and a threshold away from the observed scores, over a value that",
        "> merely fits these six questions.",
        "",
    ]
    path = out_dir / "calibration.md"
    path.write_text("\n".join(md))
    return path


# ------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the abstention rule.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("-k", type=int, default=None, help="chunks to retrieve (default TOP_K)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # Guard against tuning on the final evaluation set by mistake.
    if args.questions.name == "questions.json":
        print("WARNING: questions.json is the FINAL evaluation set. Tuning on it "
              "makes the final scores optimistic. Use calibration.json.\n")

    # Same settings as the real pipeline (.env), so calibration matches it.
    config = RetrievalConfig.from_env()
    # -k on the command line overrides TOP_K from .env.
    k = args.k or config.top_k
    # List of question dicts: id, question, expected_sources, required_terms, should_abstain.
    cases = json.loads(args.questions.read_text())

    # Always rebuild, so calibration never runs against a stale saved index.
    store = build_vector_store(config)
    # Retrieve + measure every question, then evaluate all candidate rules.
    rows = run_calibration(cases, store, k)
    sweeps = sweep(rows)
    gap = score_gap(rows)
    summary = retrieval_summary(rows, k)

    # Console output: the per-question table and the most useful sweep results.
    print(f"model={config.embedding_model}@{config.embedding_revision} k={k}\n")
    print(rows_table(rows), "\n")
    print("Retrieval (supported cases):", summary)
    print("Score separation:", gap, "\n")
    print("Rule A (score threshold):")
    print(sweep_table(sweeps["A_score"], ["min_score"]), "\n")
    for name, columns in [("B_score_margin", ["min_score", "min_margin"]),
                          ("C_score_overlap", ["min_score", "min_overlap"])]:
        print(f"Best {name}:")
        print(sweep_table(best(sweeps[name])[:8], columns), "\n")

    path = write_report(rows, sweeps, gap, summary, config, k, args.questions, args.out_dir)
    print(f"Saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
          f"and calibration.json")


# Run main() only when executed as a script (python -m src.calibration).
if __name__ == "__main__":
    main()
