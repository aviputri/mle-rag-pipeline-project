"""
Load the Northstar knowledge base and split it into retrieval chunks.

Workflows:

* Markdown files: ``document_id`` (e.g. ``billing-refunds``) is cited in answers.
* ``# `` heading = the document title.
* ``##``/``###`` headings (in future documents) = ``section`` of the chunks below them.
* Paragraphs (separated by blank lines) = the natural unit
  (each paragraph states one policy point, so we do not cut unless > ``max_chars``)
* One paragraph = one chunk
  A paragraph < ``min_chars`` --> merged with the next one
* An oversized paragraph is split at sentence boundaries.
  ``overlap_sentences`` is carried into the next piece.
* Hard line-wraps inside a paragraph are joined with spaces.

``uv run python -m src.ingestion``
  --> chunk-count summary and a sample.

"""

# Allows "str | None" style type hints to be used everywhere without runtime cost.
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------- settings

# <repo>/data/knowledge_base, resolved from this file so the module works
# no matter which folder you run it from.
DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
# A paragraph longer than this is split at sentence boundaries.
DEFAULT_MAX_CHARS = 500
# A paragraph shorter than this is merged with the next one (no tiny chunks).
DEFAULT_MIN_CHARS = 150
# Number of sentences repeated at the start of the next piece of a long paragraph.
DEFAULT_OVERLAP_SENTENCES = 1

# Matches a Markdown heading line: group(1) = the #'s (level), group(2) = heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Sentence boundary: whitespace after . ! or ? that is followed by a capital letter,
# digit, backtick, quote or bracket. Splitting on it keeps the punctuation.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])")


# ------------------------------------------------------------- data shapes


# One source file as loaded from disk (before chunking).
# frozen=True makes instances read-only, so nothing can change them by accident.
@dataclass(frozen=True)
class Document:
    document_id: str  # file name without .md, e.g. "billing-refunds"
    title: str        # text of the "# " heading
    source_path: str  # repo-relative path, e.g. "data/knowledge_base/billing-refunds.md"
    text: str         # full raw Markdown, unchanged
    sha256: str       # fingerprint of the raw text, changes if the file is edited


# One retrieval unit. Every field ends up as metadata next to its vector in Qdrant.
@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # "<document_id>#<chunk_index>", unique and stable
    document_id: str     # cited in answers as [document_id]
    title: str
    section: str | None  # nearest ## heading, None if the doc has none
    chunk_index: int     # 0-based position within the document
    text: str            # what the generator sees and what we quote
    source_path: str     # repo-relative path to the source file
    doc_sha256: str      # lets you detect a stale index after a source edit

    # @property: use it like an attribute (chunk.embedding_text), no brackets.
    @property
    def embedding_text(self) -> str:
        """Text to embed: the title adds topic context to short paragraphs."""
        # "Title - Section" when there is a section, otherwise just "Title".
        prefix = f"{self.title} - {self.section}" if self.section else self.title
        return f"{prefix}\n\n{self.text}"

    def to_payload(self) -> dict:
        """Metadata stored next to the vector (e.g. as a Qdrant payload)."""
        # asdict turns the dataclass into a plain {field_name: value} dict.
        return asdict(self)


# ----------------------------------------------------------------- loading


def load_documents(kb_dir: Path = DEFAULT_KB_DIR) -> list[Document]:
    """Read every Markdown source, unchanged, in a deterministic order."""
    # sorted() so chunk order (and therefore chunk IDs) is the same on every machine.
    paths = sorted(Path(kb_dir).glob("*.md"))
    # Fail loudly instead of silently building an empty index.
    if not paths:
        raise FileNotFoundError(f"No Markdown files found in {kb_dir}")

    documents = []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                # path.stem = file name without extension -> the citation ID.
                document_id=path.stem,
                # Fall back to the file name if a file has no "# " heading.
                title=_extract_title(raw, fallback=path.stem),
                source_path=_relative(path),
                text=raw,
                sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            )
        )
    return documents


# ---------------------------------------------------------------- chunking


def chunk_document(
    doc: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Split one document into paragraph-aligned chunks with metadata."""
    # Collect (section, text) pairs first; turn them into Chunk objects at the end.
    pieces: list[tuple[str | None, str]] = []  # (section, text)
    for section, paragraphs in _sections(doc.text):
        # Holds short paragraph(s) waiting to be merged with the next one.
        buffer = ""
        for para in paragraphs:
            # Case 1: paragraph too long -> split it into sentence-based pieces.
            if len(para) > max_chars:
                # Emit whatever short text was waiting so it isn't lost.
                if buffer:
                    pieces.append((section, buffer))
                    buffer = ""
                for part in _split_long(para, max_chars, overlap_sentences):
                    pieces.append((section, part))
            # Case 2: normal paragraph -> add it to the buffer ...
            else:
                buffer = f"{buffer}\n\n{para}" if buffer else para
                # ... and emit the buffer as soon as it is long enough.
                # A normal paragraph (>= min_chars) is therefore emitted on its own.
                if len(buffer) >= min_chars:
                    pieces.append((section, buffer))
                    buffer = ""
        # A short paragraph left over at the end of a section.
        if buffer:  # trailing short paragraph: attach to previous chunk
            if pieces and pieces[-1][0] == section:
                pieces[-1] = (section, f"{pieces[-1][1]}\n\n{buffer}")
            # Nothing to attach to (the section has only this short text): keep it.
            else:
                pieces.append((section, buffer))

    # Number the pieces and attach the document's metadata to each one.
    return [
        Chunk(
            chunk_id=f"{doc.document_id}#{i}",
            document_id=doc.document_id,
            title=doc.title,
            section=section,
            chunk_index=i,
            text=text,
            source_path=doc.source_path,
            doc_sha256=doc.sha256,
        )
        for i, (section, text) in enumerate(pieces)
    ]


def build_chunks(
    kb_dir: Path = DEFAULT_KB_DIR,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Load and chunk the whole knowledge base."""
    # Main entry point used by retrieval.py: one flat list of chunks for all docs.
    chunks: list[Chunk] = []
    for doc in load_documents(kb_dir):
        chunks.extend(chunk_document(doc, max_chars, overlap_sentences, min_chars))
    return chunks


# --------------------------------------------------------------------- helpers
# (A leading underscore means "internal to this module".)


def _relative(path: Path) -> str:
    """Repo-relative path, so payloads don't leak machine-specific paths."""
    # Repository root = the folder above src/.
    root = Path(__file__).resolve().parent.parent
    try:
        # as_posix() uses "/" separators on every operating system.
        return path.resolve().relative_to(root).as_posix()
    # The file is outside the repo (e.g. a custom --kb-dir): keep the path as given.
    except ValueError:
        return path.as_posix()


def _extract_title(raw: str, fallback: str) -> str:
    # Return the text of the first level-1 heading ("# Title").
    for line in raw.splitlines():
        match = _HEADING_RE.match(line)
        # group(1) is the run of '#'; length 1 means a level-1 heading.
        if match and len(match.group(1)) == 1:
            return match.group(2)
    return fallback


def _sections(raw: str) -> list[tuple[str | None, list[str]]]:
    """Group paragraphs under their nearest ## heading (H1 is the title)."""
    # Start with an unnamed section for paragraphs that come before any "##".
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    # Lines of the paragraph currently being read.
    current: list[str] = []

    # Finish the current paragraph: join its lines with spaces and store it
    # in the most recent section. Called at blank lines and headings.
    def flush() -> None:
        if current:
            sections[-1][1].append(" ".join(current))
            current.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        heading = _HEADING_RE.match(stripped)
        # A heading ends the paragraph before it.
        if heading:
            flush()
            # The "# " title is skipped here (it is already in Chunk.title).
            if len(heading.group(1)) > 1:  # ## or deeper starts a section
                sections.append((heading.group(2), []))
        # A blank line ends a paragraph.
        elif not stripped:
            flush()
        # Any other line belongs to the current paragraph.
        else:
            current.append(stripped)  # join hard-wrapped lines
    # The file may not end with a blank line, so store the last paragraph too.
    flush()
    # Drop sections that ended up with no paragraphs (e.g. the unnamed one).
    return [(name, paras) for name, paras in sections if paras]


def _split_long(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    """Split at sentence boundaries, repeating `overlap` sentences."""
    sentences = _SENTENCE_RE.split(paragraph)
    parts: list[str] = []
    # Sentences of the piece currently being built.
    window: list[str] = []
    for sentence in sentences:
        # Adding this sentence would exceed the limit -> close the current piece.
        if window and len(" ".join(window + [sentence])) > max_chars:
            parts.append(" ".join(window))
            # Start the next piece with the last `overlap` sentence(s).
            window = window[-overlap:] if overlap else []
            # drop the overlap if it would push this piece over the limit
            if window and len(" ".join(window + [sentence])) > max_chars:
                window = []
        window.append(sentence)
    # Store the final piece.
    if window:
        parts.append(" ".join(window))
    return parts


# ------------------------------------------------------------------------- CLI


def main() -> None:
    # Command-line options; each default comes from the settings at the top.
    parser = argparse.ArgumentParser(description="Chunk the knowledge base.")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_SENTENCES)
    parser.add_argument("--sample", default="billing-refunds#0",
                        help="chunk_id to print in full")
    args = parser.parse_args()

    chunks = build_chunks(args.kb_dir, args.max_chars, args.overlap, args.min_chars)
    # Count chunks per document for the summary table.
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1

    print(f"{len(counts)} documents -> {len(chunks)} chunks "
          f"(max_chars={args.max_chars}, min_chars={args.min_chars}, "
          f"overlap={args.overlap} sentence)\n")
    for doc_id, n in counts.items():
        print(f"  {doc_id:<28} {n}")

    # Show the requested chunk with all its metadata; fall back to the first
    # chunk if that ID does not exist.
    sample = next((c for c in chunks if c.chunk_id == args.sample), chunks[0])
    print("\nSample chunk:")
    for key, value in sample.to_payload().items():
        print(f"  {key}: {value}")


# Run main() only when executed as a script (python -m src.ingestion),
# not when another module imports this file.
if __name__ == "__main__":
    main()
