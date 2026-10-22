"""Evaluation rubrics behave as documented in src/evaluation.py."""

from src.evaluation import (
    evaluate_case,
    faithfulness,
    judge_sentence,
    retrieval_metrics,
    split_sentences,
    summarise,
    term_coverage,
)
from src.pipeline import RAGAnswer

SOURCE = ("Northstar refunds duplicate payments when two captured transactions correspond "
          "to the same invoice. The requester must provide the invoice number and both "
          "payment references within 30 calendar days of the second charge.")


def test_split_sentences_keeps_citation_with_its_sentence():
    parts = split_sentences("First fact. [a] Second fact [b]. Third. [c]")
    assert parts == ["First fact [a].", "Second fact [b].", "Third [c]."]


def test_verbatim_sentence_is_literal():
    s = "The requester must provide the invoice number and both payment references. [billing-refunds]"
    assert judge_sentence(s, SOURCE)["label"] == "literal"


def test_close_paraphrase_is_supported():
    s = "You must provide the invoice number and both payment references [billing-refunds]."
    assert judge_sentence(s, SOURCE)["label"] == "supported"


def test_wrong_number_is_unsupported():
    # Same words, invented number: must not count as faithful.
    s = "Provide the invoice number and payment references within 60 calendar days."
    verdict = judge_sentence(s, SOURCE)
    assert verdict["label"] == "unsupported" and not verdict["numbers_ok"]


def test_unrelated_claim_is_unsupported():
    assert judge_sentence("Students receive a 50 percent discount.", SOURCE)["label"] == "unsupported"


def _answer(text, context_ids, citations):
    return RAGAnswer(question="q", status="answered", answer=text,
                     citations=citations, context_ids=context_ids)


def test_faithfulness_uses_only_cited_documents():
    chunks = {"billing-refunds#0": SOURCE,
              "invoice-corrections#0": "A correction request must include the invoice number."}
    # Claim is in invoice-corrections, but the sentence cites billing-refunds.
    result = _answer("A correction request must include the invoice number. [billing-refunds]",
                     list(chunks), ["billing-refunds"])
    assert faithfulness(result, chunks)["score"] < 1.0


def test_faithfulness_score_is_share_of_supported_sentences():
    chunks = {"billing-refunds#0": SOURCE}
    result = _answer(f"{SOURCE.split('. ')[0]}. [billing-refunds] Students get a discount. [billing-refunds]",
                     list(chunks), ["billing-refunds"])
    verdict = faithfulness(result, chunks)
    assert verdict["score"] == 0.5 and not verdict["faithful"]


def test_term_coverage():
    assert term_coverage("Give the Invoice Number.", ["invoice number", "payment references"]) == 0.5
    assert term_coverage("anything", []) is None


def test_retrieval_metrics_multi_source():
    m = retrieval_metrics(["privacy-data", "account-mfa", "privacy-data"], ["account-mfa", "technical-incidents"])
    assert m == {"hit": 1, "recall": 0.5, "reciprocal_rank": 0.5, "first_rank": 2}
    assert retrieval_metrics(["a", "b"], ["c"])["reciprocal_rank"] == 0.0


def test_decision_and_summary():
    supported = {"id": "s", "question": "q", "should_abstain": False,
                 "expected_sources": ["billing-refunds"], "required_terms": ["invoice number"]}
    unsupported = {"id": "u", "question": "q", "should_abstain": True,
                   "expected_sources": [], "required_terms": []}
    chunks = {"billing-refunds#0": SOURCE}
    answered = RAGAnswer(question="q", status="answered",
                         answer="Provide the invoice number. [billing-refunds]",
                         citations=["billing-refunds"], context_ids=list(chunks),
                         retrieved=[{"rank": 1, "chunk_id": "billing-refunds#0",
                                     "document_id": "billing-refunds", "score": 0.6}])
    abstained = RAGAnswer(question="q", status="abstained", answer="x")
    rows = [evaluate_case(supported, answered, chunks), evaluate_case(unsupported, abstained, chunks)]
    s = summarise(rows)
    assert s["decision_accuracy"] == 1.0 and s["false_answers"] == 0
    assert s["answer_relevance"] == 1.0 and s["mrr"] == 1.0 and s["citation_precision"] == 1.0


def test_typographic_characters_do_not_break_matching():
    # Regression: gpt-oss wrote "30 minutes" and "lost‑device".
    assert term_coverage("Updates every 30 minutes.", ["30 minutes"]) == 1.0
    source = "describe the lost-device situation"
    assert judge_sentence("Describe the lost‑device situation.", source)["label"] == "literal"
