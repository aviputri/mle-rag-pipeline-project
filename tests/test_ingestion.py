"""Ingestion: documents load unchanged and every chunk keeps citable metadata."""

import re
from pathlib import Path

from src.ingestion import DEFAULT_KB_DIR, build_chunks, chunk_document, load_documents

KB_FILES = sorted(p.stem for p in Path(DEFAULT_KB_DIR).glob("*.md"))


def test_every_source_document_is_loaded():
    docs = load_documents()
    # One Document per Markdown file, ID = file name without .md.
    assert [d.document_id for d in docs] == KB_FILES
    assert len(docs) == 10
    for doc in docs:
        assert doc.title and not doc.title.startswith("#")
        assert len(doc.sha256) == 64
        assert doc.source_path.startswith("data/knowledge_base/")


def test_chunks_keep_required_metadata(chunks):
    # Project requirement: document_id, title and chunk position on every chunk.
    for chunk in chunks:
        assert chunk.document_id in KB_FILES
        assert chunk.title
        assert chunk.chunk_id == f"{chunk.document_id}#{chunk.chunk_index}"
        assert chunk.text.strip()
        payload = chunk.to_payload()
        for key in ("chunk_id", "document_id", "title", "chunk_index", "text", "source_path"):
            assert key in payload


def test_chunk_ids_are_unique_and_positions_consecutive(chunks):
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # Within each document the positions are 0, 1, 2, ... with no gaps.
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c.chunk_index)
    for indices in by_doc.values():
        assert indices == list(range(len(indices)))


def test_no_source_text_is_lost(chunks):
    # Every word sequence of every paragraph must appear in some chunk.
    joined = " ".join(c.text for c in chunks)
    for doc in load_documents():
        body = re.sub(r"^# .*$", "", doc.text, flags=re.MULTILINE)
        for paragraph in re.split(r"\n\s*\n", body):
            flat = " ".join(paragraph.split())
            if flat:
                assert flat in joined, f"lost text from {doc.document_id}: {flat[:60]}"


def test_chunking_is_deterministic():
    assert build_chunks() == build_chunks()


def test_malicious_example_sits_in_its_own_chunk(chunks):
    # The guardrail can then drop it without discarding legitimate guidance.
    bad = [c for c in chunks if "IGNORE ALL PRIOR INSTRUCTIONS" in c.text]
    assert len(bad) == 1
    assert "When suspicious instructions appear" not in bad[0].text


def test_embedding_text_adds_title_but_text_stays_plain(chunks):
    chunk = chunks[0]
    assert chunk.embedding_text.startswith(chunk.title)
    assert not chunk.text.startswith(chunk.title)


def _doc_from(tmp_path, text):
    (tmp_path / "sample.md").write_text(text)
    return load_documents(tmp_path)[0]


def test_long_paragraph_is_split_at_sentences_with_overlap(tmp_path):
    sentences = [f"Sentence number {i} explains one rule in enough words." for i in range(12)]
    doc = _doc_from(tmp_path, "# Sample\n\n" + " ".join(sentences) + "\n")
    parts = chunk_document(doc, max_chars=200, overlap_sentences=1)
    assert len(parts) > 1
    assert all(len(p.text) <= 200 for p in parts)
    # Each piece ends with a full sentence and the next piece repeats it.
    for left, right in zip(parts, parts[1:]):
        assert left.text.endswith(".")
        assert right.text.startswith(left.text.split(". ")[-1].rstrip("."))


def test_short_trailing_paragraph_is_merged_backwards(tmp_path):
    long_para = "This paragraph is long enough to stand alone as a chunk. " * 4
    doc = _doc_from(tmp_path, f"# Sample\n\n{long_para}\n\nShort tail.\n")
    parts = chunk_document(doc, min_chars=150)
    assert len(parts) == 1
    assert parts[0].text.endswith("Short tail.")


def test_subheadings_become_sections(tmp_path):
    para = "Some policy text that is long enough to be kept as its own chunk here. " * 3
    doc = _doc_from(tmp_path, f"# Sample\n\n## Eligibility\n\n{para}\n\n## Process\n\n{para}\n")
    parts = chunk_document(doc)
    assert [p.section for p in parts] == ["Eligibility", "Process"]
    assert parts[0].embedding_text.startswith("Sample - Eligibility")
