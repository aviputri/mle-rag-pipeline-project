"""Shared test fixtures. All tests run offline: no model download, no API key.

pytest loads this file automatically and makes every @pytest.fixture in it
available to all test files by parameter name (e.g. ``def test_x(chunks):``).
"""

import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings

from src.calibration import content_words
from src.ingestion import build_chunks
from src.retrieval import RetrievalConfig, RetrievedChunk, build_vector_store


class KeywordEmbeddings(Embeddings):
    """Tiny offline stand-in for MiniLM: a hashed bag of word stems.

    Each content word is cut to its first 5 letters (refund/refunds -> "refun"),
    hashed into one of DIM slots, counted, then L2-normalised. Texts that share
    words get a high cosine score, so retrieval tests behave sensibly without
    downloading a real model. Deterministic: same text -> same vector.
    """

    DIM = 512

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for word in content_words(text):
            # md5 instead of hash(): Python's hash() changes between runs.
            slot = int(hashlib.md5(word[:5].encode()).hexdigest(), 16) % self.DIM
            vector[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    # The two methods every LangChain Embeddings class must implement.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# scope="session": build once and share across all tests (faster).
@pytest.fixture(scope="session")
def chunks():
    return build_chunks()


@pytest.fixture(scope="session")
def chunk_by_id(chunks):
    return {c.chunk_id: c for c in chunks}


@pytest.fixture(scope="session")
def test_config():
    # In-memory store (storage_path=None) so tests never touch storage/qdrant.
    return RetrievalConfig(embedding_model="keyword-test", embedding_revision=None,
                           top_k=3, storage_path=None)


@pytest.fixture(scope="session")
def store(chunks, test_config):
    return build_vector_store(test_config, chunks=chunks, embeddings=KeywordEmbeddings())


@pytest.fixture
def make_hits(chunk_by_id):
    """Build RetrievedChunk lists by hand: make_hits([("billing-refunds#0", 0.5), ...]).

    Lets tests replay exact retrieval results (e.g. the real MiniLM scores
    from calibration) without running a model.
    """

    def _make(pairs):
        hits = []
        for rank, (chunk_id, score) in enumerate(pairs, start=1):
            chunk = chunk_by_id[chunk_id]
            hits.append(RetrievedChunk(
                rank=rank, score=score, chunk_id=chunk_id,
                document_id=chunk.document_id, title=chunk.title,
                text=chunk.text, metadata=chunk.to_payload(),
            ))
        return hits

    return _make


# Real top-3 MiniLM results for calibration questions (from src.calibration, k=3).
CAL01_QUESTION = "Which payment details must I include when reporting a duplicate charge?"
CAL01_HITS = [("invoice-corrections#1", 0.550), ("billing-refunds#2", 0.533), ("billing-refunds#0", 0.446)]
CAL05_QUESTION = "What is the cancellation refund policy for an annual subscription?"
CAL05_HITS = [("billing-refunds#2", 0.441), ("billing-refunds#0", 0.392), ("invoice-corrections#1", 0.361)]
