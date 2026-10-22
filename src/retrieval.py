"""
Embed knowledge-base chunks and retrieve them with LangChain + Qdrant.

Workflows:

* Embeddings: 
  ``sentence-transformers/all-MiniLM-L6-v2`` (384 dims) through
  ``langchain_huggingface.HuggingFaceEmbeddings`` 
  pinned to a Hugging Face revision so every rebuild produces the same vectors. 
  Runs on CPU, no key.
  Vectors are L2-normalised, so cosine similarity = dot product.
* Vector store: 
  Qdrant in *local mode* through ``langchain_qdrant``. 
  No Docker or server. 
  ``VECTOR_STORAGE_PATH`` empty -> in-memory collection rebuilt by
  every command; set -> on-disk collection under that folder.
* What is embedded: ``Chunk.embedding_text`` (title + paragraph). 
  The plain paragraph text and all chunk metadata live in ``Document.metadata``
  so a hit can be cited as ``[document_id]`` and quoted without the title prefix.
* Point IDs are UUIDv5 of ``chunk_id``
  so rebuilding replaces points instead of duplicating them.
* Scores: Qdrant cosine similarity, higher = more similar (range -1..1).
  Thresholding/abstention is applied later in the pipeline
  this module always returns the raw top-k for evaluation (recall@k, MRR).


``uv run python -m src.retrieval rebuild``
``uv run python -m src.retrieval search "How do I report a duplicate charge?"``
``uv run python -m src.retrieval search "Which payment details must I include when reporting a duplicate charge?"``
"""

# Allows "str | None" style type hints to be used everywhere without runtime cost.
from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
# LangChain's standard container: page_content (text) + metadata (dict).
from langchain_core.documents import Document
# Base class every LangChain embedding model implements (used for type hints).
from langchain_core.embeddings import Embeddings
# Runs a sentence-transformers model locally behind LangChain's interface.
from langchain_huggingface import HuggingFaceEmbeddings
# LangChain wrapper around a Qdrant collection (embeds, stores and searches).
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance

from src.ingestion import Chunk, build_chunks

# Read settings (EMBEDDING_MODEL, TOP_K, VECTOR_STORAGE_PATH, ...) from .env
# into environment variables. Values already set in the shell take priority.
load_dotenv()

# ---------------------------------------------------------------- settings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Exact model commit on Hugging Face (same as the course notebooks), so the
# vectors are identical on every machine and after every rebuild.
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# Name of the Qdrant collection (like a table name in a database).
COLLECTION_NAME = "northstar_kb"
# Cosine similarity: compares vector direction, the usual choice for text embeddings.
DISTANCE = Distance.COSINE


# All retrieval settings in one read-only object.
@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str
    embedding_revision: str | None
    top_k: int
    storage_path: str | None  # None -> in-memory

    # Alternative constructor: RetrievalConfig.from_env() builds the config
    # from environment variables, with defaults for anything left empty.
    @classmethod
    def from_env(cls) -> RetrievalConfig:
        # "or" also catches empty strings such as "EMBEDDING_MODEL=" in .env.
        model = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        # Use the pinned revision only for the default model; another model
        # needs its own EMBEDDING_REVISION (or None = latest).
        revision = os.getenv("EMBEDDING_REVISION") or (
            DEFAULT_EMBEDDING_REVISION if model == DEFAULT_EMBEDDING_MODEL else None
        )
        return cls(
            embedding_model=model,
            embedding_revision=revision,
            top_k=int(os.getenv("TOP_K") or 3),
            storage_path=os.getenv("VECTOR_STORAGE_PATH") or None,
        )


# One search hit, with the fields the rest of the pipeline needs.
@dataclass(frozen=True)
class RetrievedChunk:
    rank: int            # 1-based
    score: float         # cosine similarity, higher is better
    chunk_id: str
    document_id: str
    title: str
    text: str            # paragraph text without the title prefix
    metadata: dict       # full chunk payload, in case you need more fields


# ------------------------------------------------------------------ embeddings


# lru_cache remembers the result per (model, revision), so the model is
# loaded from disk only once even if get_embeddings() is called many times.
@lru_cache(maxsize=4)
def get_embeddings(model: str, revision: str | None) -> HuggingFaceEmbeddings:
    """Load the embedding model once per process (it takes a few seconds)."""
    # These kwargs are passed on to sentence-transformers' SentenceTransformer.
    model_kwargs: dict = {"device": "cpu"}
    if revision:
        model_kwargs["revision"] = revision
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs,
        # Unit-length vectors: cosine similarity becomes a plain dot product
        # and scores are comparable across queries.
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------- vector store


def chunk_to_document(chunk: Chunk) -> Document:
    # page_content is what gets embedded (title + paragraph); metadata keeps
    # every chunk field, including the plain text used for citing/quoting.
    return Document(page_content=chunk.embedding_text, metadata=chunk.to_payload())


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point ID derived from the chunk ID."""
    # Qdrant IDs must be an integer or a UUID. uuid5 always gives the same UUID
    # for the same input, so "billing-refunds#0" always maps to the same point.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"northstar/{chunk_id}"))


def build_vector_store(
    config: RetrievalConfig | None = None,
    chunks: list[Chunk] | None = None,
    embeddings: Embeddings | None = None,
) -> QdrantVectorStore:
    """(Re)build the collection from the source documents.

    ``force_recreate=True`` drops the old collection first,
    so an edited or deleted source can never leave stale chunks behind.
    Pass ``embeddings`` to swap the model.
    """
    config = config or RetrievalConfig.from_env()
    # By default, chunk the knowledge base fresh from the Markdown files.
    chunks = chunks if chunks is not None else build_chunks()
    documents = [chunk_to_document(c) for c in chunks]
    # On disk if a storage path is configured, otherwise in RAM only.
    location = {"path": config.storage_path} if config.storage_path else {"location": ":memory:"}
    # One call that creates the collection, embeds every document and stores
    # vector + metadata as Qdrant points.
    return QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings
        or get_embeddings(config.embedding_model, config.embedding_revision),
        ids=[point_id(c.chunk_id) for c in chunks],
        collection_name=COLLECTION_NAME,
        distance=DISTANCE,
        force_recreate=True,
        # ** unpacks the dict into keyword arguments: path=... or location=...
        **location,
    )


def load_vector_store(config: RetrievalConfig | None = None) -> QdrantVectorStore:
    """Open a persisted collection, or build an in-memory one."""
    config = config or RetrievalConfig.from_env()
    # An in-memory store does not survive between runs, so build it now.
    if not config.storage_path:
        return build_vector_store(config)
    # Reuse vectors saved by an earlier "rebuild" (no re-embedding needed).
    # The same embedding model is still needed to embed incoming queries.
    return QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(config.embedding_model, config.embedding_revision),
        path=config.storage_path,
    )


# ------------------------------------------------------------------- retrieval


def retrieve(store: QdrantVectorStore, query: str, k: int = 3) -> list[RetrievedChunk]:
    """Return the top-k chunks with their similarity scores, best first."""
    # Embeds the query with the store's model and returns [(Document, score), ...]
    # sorted by score, highest first.
    hits = store.similarity_search_with_score(query, k=k)
    return [
        RetrievedChunk(
            rank=rank,
            score=float(score),
            chunk_id=doc.metadata["chunk_id"],
            document_id=doc.metadata["document_id"],
            title=doc.metadata["title"],
            # Plain paragraph from metadata, not page_content (which has the title prefix).
            text=doc.metadata["text"],
            metadata=doc.metadata,
        )
        # enumerate(..., start=1) numbers the hits 1, 2, 3, ...
        for rank, (doc, score) in enumerate(hits, start=1)
    ]


# ------------------------------------------------------------------------- CLI


def main() -> None:
    # Two subcommands: "rebuild" and "search <query> [-k N]".
    parser = argparse.ArgumentParser(description="Build and query the vector index.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rebuild", help="re-embed all sources into a fresh collection")
    search = sub.add_parser("search", help="show the top-k chunks for a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=None)
    args = parser.parse_args()

    config = RetrievalConfig.from_env()
    # Print the configuration so every run documents what it used.
    print(f"model={config.embedding_model}@{config.embedding_revision} "
          f"distance={DISTANCE.value} collection={COLLECTION_NAME} "
          f"storage={config.storage_path or ':memory:'}")

    if args.command == "rebuild":
        store = build_vector_store(config)
        # Ask Qdrant how many points the collection now holds.
        count = store.client.count(COLLECTION_NAME).count
        print(f"Indexed {count} chunks.")
        return

    # "search": in-memory -> build first; on disk -> open the saved collection.
    store = build_vector_store(config) if not config.storage_path else load_vector_store(config)
    # -k on the command line overrides TOP_K from .env.
    for hit in retrieve(store, args.query, k=args.k or config.top_k):
        print(f"\n#{hit.rank}  score={hit.score:.3f}  [{hit.document_id}]  {hit.chunk_id}")
        print(f"    {hit.text}")


# Run main() only when executed as a script (python -m src.retrieval),
# not when another module imports this file.
if __name__ == "__main__":
    main()
