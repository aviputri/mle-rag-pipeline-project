"""Retrieval: the index holds every chunk, search returns ranked, citable hits."""

import pytest

from src.retrieval import (
    COLLECTION_NAME,
    RetrievalConfig,
    build_vector_store,
    point_id,
    retrieve,
)
from tests.conftest import KeywordEmbeddings


def test_index_contains_every_chunk(store, chunks):
    assert store.client.count(COLLECTION_NAME, exact=True).count == len(chunks)


def test_retrieve_returns_ranked_hits_with_metadata(store, chunk_by_id):
    hits = retrieve(store, "How long is a password reset link valid?", k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    # Best first: scores never increase down the list.
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))
    for hit in hits:
        chunk = chunk_by_id[hit.chunk_id]
        assert hit.document_id == chunk.document_id
        # Plain paragraph text (no title prefix) for quoting and citing.
        assert hit.text == chunk.text


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How long is a password reset link valid?", "account-password"),
        ("A client receives 429 responses. How should it retry the API?", "technical-api-limits"),
        ("Can finance correct the billing address on an issued invoice?", "invoice-corrections"),
    ],
)
def test_known_questions_retrieve_expected_document(store, question, expected):
    # Uses the keyword stand-in model; real MiniLM scores are in reports/calibration.
    top_docs = [h.document_id for h in retrieve(store, question, k=3)]
    assert expected in top_docs


def test_k_controls_number_of_hits(store):
    assert len(retrieve(store, "refund", k=1)) == 1
    assert len(retrieve(store, "refund", k=5)) == 5


def test_point_ids_are_stable_and_distinct():
    assert point_id("billing-refunds#0") == point_id("billing-refunds#0")
    assert point_id("billing-refunds#0") != point_id("billing-refunds#1")


def test_rebuild_on_disk_does_not_duplicate(tmp_path, chunks):
    config = RetrievalConfig("keyword-test", None, 3, str(tmp_path / "qdrant"))
    for _ in range(2):
        store = build_vector_store(config, chunks=chunks, embeddings=KeywordEmbeddings())
        count = store.client.count(COLLECTION_NAME, exact=True).count
        store.client.close()  # release the local-mode folder lock
    assert count == len(chunks)


def test_config_reads_env_and_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("EMBEDDING_REVISION", "")
    monkeypatch.setenv("VECTOR_STORAGE_PATH", "")
    config = RetrievalConfig.from_env()
    assert config.top_k == 5
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.embedding_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert config.storage_path is None
