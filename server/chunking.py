"""Chunking-Logik: Markdown-Text in (formelschonende) Chunks zerlegen.

Chunking-Methoden, unterscheidbar über das Enum ChunkingMethod:

* RECURSIVE  – Fixed-Size Chunking mit Overlap (Chonkie TokenChunker), token-basiert;
               gleichmäßige Chunks am Token-Budget des Embedding-Modells.
* MARKDOWN   – Document-Based/Structural Chunking (Chonkie RecursiveChunker): trennt
               hierarchisch bevorzugt an Überschriften/Absätzen/Sätzen und packt bis
               zur Chunk-Größe (KEINE h1/h2/h3-Metadaten).
* SEMANTIC   – Semantic Chunking (Chonkie SemanticChunker): trennt an semantischen
               Ähnlichkeitsgrenzen zwischen Sätzen.
* LATE       – Late Chunking mit Chonkie: der (lange) Text wird ZUERST embeddet
               (Token-Level), dann werden Chunk-Grenzen gelegt und die Token-Vektoren
               PRO Chunk gemittelt. Jeder Chunk-Vektor trägt so Dokumentkontext.
               Anders als die anderen Methoden liefert LATE bereits die Embeddings
               mit (ChunkRecord.embedding); diese werden in einer eigenen Collection
               gespeichert (siehe database.py), weil ihre Dimension vom Standard-
               Embedding-Modell abweicht.

Alle Methoden nutzen ausschließlich Chonkie (kein LangChain mehr).
Die (text-basierten) Methoden schützen LaTeX-Formeln vor dem Zerschneiden (siehe
Formel-Schutz). Jeder Chunk wird mit metadata['method'] markiert, damit die DB
gezielt nach den Chunks einer bestimmten Methode suchen kann.
"""

import server.hf_offline  # noqa: F401 -- MUSS zuerst stehen: HF-Offline vor HF-nutzenden Imports

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from chonkie import RecursiveChunker, SemanticChunker, TokenChunker
from chonkie.types.recursive import RecursiveLevel, RecursiveRules


# ========= CHUNKING-METHODEN =========

class ChunkingMethod(str, Enum):
    """Verfügbare Chunking-Methoden. Der Wert (str) wird als Metadatum gespeichert
    und über die API übertragen; das Notebook spiegelt dieses Enum clientseitig."""

    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"  # Semantic Chunking (Chonkie)
    LATE = "late"          # Late Chunking (Chonkie) – liefert Embeddings mit

    @classmethod
    def from_value(cls, value: "str | ChunkingMethod | None") -> "ChunkingMethod":
        """Robuste Umwandlung eines eingehenden Werts (z. B. aus der API) in das Enum."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Unbekannte Chunking-Methode: {value!r}. Erlaubt: {valid}"
            ) from exc


# ========= GEMEINSAMES CHUNK-FORMAT =========

@dataclass
class ChunkRecord:
    """Ein Chunk unabhängig von der Methode.

    embedding ist nur bei Methoden gesetzt, die selbst embedden (LATE); bei den
    text-basierten Methoden bleibt es None und die DB embeddet den Text selbst.
    """

    text: str
    metadata: dict
    embedding: "list[float] | None" = None


# ========= CHUNKER =========

# Gemeinsames Embedding-Modell für ALLE Methoden (Option A): ein einziger Vektorraum,
# damit die Retrieval-Distanzen über alle Methoden vergleichbar sind und eine gemeinsame,
# faire Schwelle möglich ist (kein Late-Sonderraum mehr). Mehrsprachig + Langkontext
# (deutschsprachiges Vorlesungsmaterial). Wird auch als Tokenizer der token-basierten
# Chunker verwendet, damit die 128-Token-Grenze im selben Tokenraum gemessen wird.
# Per .env überschreibbar.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# RECURSIVE = Fixed-Size Chunking mit Overlap (Chonkie TokenChunker), token-basiert.
# chunk_size in Tokens; 128 als bewusst gewählte, für alle Methoden gleiche Granularität
# (e5-base könnte mehr, aber die feste Größe hält den Methodenvergleich sauber).
# chunk_overlap (16) >= Länge der Formel-Platzhalter, damit eine maskierte Formel nie
# über eine Chunk-Grenze verloren geht (sie erscheint dann ganz im Überlappungs-Chunk).
_recursive_splitter = TokenChunker(
    tokenizer=EMBEDDING_MODEL,
    chunk_size=128,
    chunk_overlap=16,
)

# MARKDOWN = Document-Based/Structural Chunking (Chonkie RecursiveChunker). Trennt
# hierarchisch bevorzugt an Überschriften, dann Absätzen/Zeilen, dann Sätzen, und
# packt bis chunk_size. Anders als der frühere LangChain-Header-Splitter werden KEINE
# h1/h2/h3-Metadaten mehr extrahiert (Chonkie liefert die Struktur nicht als Metadaten).
_md_rules = RecursiveRules(
    levels=[
        RecursiveLevel(delimiters=None, pattern=r"\n(?=#{1,6}\s)", pattern_mode="split"),  # Überschriften
        RecursiveLevel(delimiters=["\n\n", "\n"]),   # Absätze / Zeilen
        RecursiveLevel(delimiters=[". ", "! ", "? "]),  # Sätze
        RecursiveLevel(whitespace=True),             # Wörter (Fallback)
    ]
)
_markdown_chunker = RecursiveChunker(
    tokenizer=EMBEDDING_MODEL,
    chunk_size=128,
    rules=_md_rules,
    min_characters_per_chunk=24,
)

_semantic_splitter = SemanticChunker(
    embedding_model=EMBEDDING_MODEL,
    threshold=0.7,
    # chunk_size an das 128-Token-Fenster des Embedding-Modells angepasst, damit
    # semantische Chunks beim Embedden nicht abgeschnitten werden.
    chunk_size=128,
    skip_window=1
)


# ========= FORMEL-SCHUTZ =========
# Formeln (LaTeX) sollen beim Chunking nicht zerschnitten werden. Dazu werden sie
# vor dem Splitten durch kurze Platzhalter ersetzt (die keine Trennzeichen des
# Splitters enthalten) und nach dem Splitten in jedem Chunk wiederhergestellt.

# Reihenfolge: Block-Formeln zuerst, damit einzelne $ danach als inline gelten.
_MATH_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),   # Block:  $$ ... $$
    re.compile(r"\\\[.*?\\\]", re.DOTALL),   # Block:  \[ ... \]
    re.compile(r"\\\(.*?\\\)", re.DOTALL),   # Inline: \( ... \)
]

# Platzhalter aus seltenen Zeichen (kein Separator der Chunker). Wird er dennoch an
# einer Chunk-Grenze getrennt, sorgt der Chunk-Overlap (>= Platzhalter-Länge) dafür,
# dass der Platzhalter im Überlappungs-Chunk ganz erscheint und dort wiederhergestellt
# wird – die Formel bleibt also erhalten.
_PLACEHOLDER = "⟦MATH{}⟧"  # ⟦MATH0⟧, ⟦MATH1⟧, ...


def _mask_math(text: str):
    """Ersetzt Formeln durch Platzhalter. Gibt (maskierter_text, formeln) zurück."""
    formulas: list[str] = []

    def _replace(match: re.Match) -> str:
        index = len(formulas)
        formulas.append(match.group(0))
        return _PLACEHOLDER.format(index)

    for pattern in _MATH_PATTERNS:
        text = pattern.sub(_replace, text)
    return text, formulas


def _unmask_math(text: str, formulas: list[str]) -> str:
    """Setzt die ursprünglichen Formeln wieder ein."""
    for index, formula in enumerate(formulas):
        text = text.replace(_PLACEHOLDER.format(index), formula)
    return text


# ========= SEITEN-ZUORDNUNG =========
# pdf_to_markdown.py stellt jeder Seite einen Marker <!-- page: N --> voran. Wir
# zerlegen den Text an diesen Markern in (Seitenzahl, Segment)-Paare, damit jeder
# Chunk seine Seite kennt (und kein Chunk über eine Seitengrenze läuft). Das Marker-
# Format muss mit pdf_to_markdown.PAGE_MARKER übereinstimmen.
_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def _split_by_page(text: str) -> "list[tuple[int | None, str]]":
    """Zerlegt Markdown an den Seiten-Markern. Ohne Marker: ein Segment mit page=None."""
    segments: list[tuple[int | None, str]] = []
    current_page: int | None = None
    pos = 0
    for match in _PAGE_MARKER_RE.finditer(text):
        segment = text[pos : match.start()]
        if segment.strip():
            segments.append((current_page, segment))
        current_page = int(match.group(1))
        pos = match.end()
    tail = text[pos:]
    if tail.strip():
        segments.append((current_page, tail))
    return segments or [(None, text)]


# ========= LATE CHUNKING (Chonkie) =========
# Late Chunking embeddet den ganzen Text zuerst und mittelt die Token-Vektoren pro
# Chunk -> jeder Chunk-Vektor kennt den Dokumentkontext. Das Embedding-Modell muss
# lang genug sein, damit der Kontext etwas bringt.
#
# Option A: Late nutzt standardmäßig DASSELBE EMBEDDING_MODEL wie die übrigen Methoden
# (ein gemeinsamer Vektorraum). Das gewählte e5-base ist mehrsprachig UND langkontext-
# fähig (512 Tokens), passt also für beide Rollen. Per .env separat überschreibbar,
# falls Late doch ein anderes Modell nutzen soll.
LATE_EMBEDDING_MODEL = os.getenv("LATE_EMBEDDING_MODEL", EMBEDDING_MODEL)
LATE_CHUNK_SIZE = int(os.getenv("LATE_CHUNK_SIZE", "512"))  # Tokens pro Chunk

_late_embeddings = None  # chonkie SentenceTransformerEmbeddings (lazy)
_late_chunker = None      # chonkie LateChunker (lazy)


def _get_late():
    """Lädt (einmalig) das Late-Embedding-Modell und den LateChunker.

    Import und Modell-Load bewusst lazy: nur wenn LATE tatsächlich genutzt wird.
    """
    global _late_embeddings, _late_chunker
    if _late_chunker is None:
        from chonkie import LateChunker
        from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings

        # Ein Modell-Objekt für Chunking UND Query-Embedding (gleicher Vektorraum).
        _late_embeddings = SentenceTransformerEmbeddings(model=LATE_EMBEDDING_MODEL)
        _late_chunker = LateChunker(
            embedding_model=_late_embeddings,
            chunk_size=LATE_CHUNK_SIZE,
        )
    return _late_embeddings, _late_chunker


def embed_query_late(query: str) -> "list[float]":
    """Embeddet eine Query mit dem Late-Modell (für die Suche in den Late-Chunks)."""
    embeddings, _ = _get_late()
    vector = embeddings.embed(query)
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def _strip_page_markers(text: str):
    """Entfernt die <!-- page: N -->-Marker und merkt sich, ab welchem (bereinigten)
    Zeichen-Offset welche Seite beginnt. Rückgabe: (clean_text, [(offset, seite), ...])."""
    boundaries: list[tuple[int, int]] = []
    parts: list[str] = []
    clean_len = 0
    pos = 0
    for match in _PAGE_MARKER_RE.finditer(text):
        segment = text[pos : match.start()]
        parts.append(segment)
        clean_len += len(segment)
        boundaries.append((clean_len, int(match.group(1))))  # ab hier: neue Seite
        pos = match.end()
    parts.append(text[pos:])
    return "".join(parts), boundaries


def _page_at(offset: int, boundaries: "list[tuple[int, int]]") -> "int | None":
    """Seitenzahl für einen Zeichen-Offset im bereinigten Text (letzte Grenze <= offset)."""
    page = None
    for start, page_no in boundaries:
        if start <= offset:
            page = page_no
        else:
            break
    return page


def pdf_source(path: str) -> str:
    """Quelle als PDF-Dateiname ableiten: die tatsächliche Quelle ist die PDF, nicht
    das daraus erzeugte Markdown. Es wird nur die Endung .md -> .pdf getauscht; der
    übrige Pfad (inkl. Trennzeichen) bleibt unverändert."""
    root, _ = os.path.splitext(path)
    return root + ".pdf"


def _late_chunks(text: str, path: str) -> "list[ChunkRecord]":
    """Late Chunking über das ganze Dokument; Seite via Zeichen-Offset zugeordnet."""
    _, chunker = _get_late()
    clean_text, boundaries = _strip_page_markers(text)

    records: list[ChunkRecord] = []
    for chunk in chunker.chunk(clean_text):
        metadata = {"source": pdf_source(path), "method": ChunkingMethod.LATE.value}
        page = _page_at(chunk.start_index, boundaries)
        if page is not None:
            metadata["page"] = page
        vector = chunk.embedding
        embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        records.append(ChunkRecord(text=chunk.text, metadata=metadata, embedding=embedding))
    return records


# ========= SPLITTER PRO METHODE (text-basiert) =========

def _recursive_chunks(masked_text: str):
    """Fixed-Size Chunks (Chonkie TokenChunker) – liefert die Chonkie-Chunks."""
    return _recursive_splitter.chunk(masked_text)


def _markdown_chunks(masked_text: str):
    """Struktur-bewusstes Splitten (Chonkie RecursiveChunker)."""
    return _markdown_chunker.chunk(masked_text)


def _semantic_chunks(masked_text: str):
    """Semantisches Splitten (Chonkie SemanticChunker)."""
    return _semantic_splitter.chunk(masked_text)


def chunk_file(
    path: str, method: "str | ChunkingMethod" = ChunkingMethod.RECURSIVE
) -> "list[ChunkRecord]":
    """Liest eine Markdown-Datei ein und zerlegt sie in Chunks.

    Args:
        path: Pfad zur Markdown-Datei.
        method: Chunking-Methode (ChunkingMethod oder deren String-Wert).

    Returns:
        Liste von ChunkRecord; jeder trägt metadata['source'] und metadata['method']
        (sowie 'page', wenn Seiten-Marker vorhanden). Bei LATE ist zusätzlich
        ChunkRecord.embedding gesetzt.
    """
    method = ChunkingMethod.from_value(method)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # LATE embeddet selbst und arbeitet auf dem ganzen Dokument (Kontext!).
    if method is ChunkingMethod.LATE:
        return _late_chunks(text, path)

    # Text-basierte Methoden: pro Seite getrennt chunken -> jeder Chunk trägt seine
    # Seitenzahl und läuft nie über eine Seitengrenze.
    records: list[ChunkRecord] = []
    for page_no, segment in _split_by_page(text):
        # Formeln vor dem Splitten schützen ...
        masked_text, formulas = _mask_math(segment)

        match method:
            case ChunkingMethod.MARKDOWN:
                chunks = _markdown_chunks(masked_text)
            case ChunkingMethod.RECURSIVE:
                chunks = _recursive_chunks(masked_text)
            case ChunkingMethod.SEMANTIC:
                chunks = _semantic_chunks(masked_text)

        # ... Formeln je Chunk wiederherstellen und als ChunkRecord sammeln.
        for chunk in chunks:
            metadata = {"source": pdf_source(path), "method": method.value}
            if page_no is not None:
                metadata["page"] = page_no
            records.append(
                ChunkRecord(text=_unmask_math(chunk.text, formulas), metadata=metadata)
            )

    return records


# ========= CLI: alle Markdown-Dateien aus data/processed chunken =========

# data/processed liegt neben dieser Datei (server/data/processed).
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"


def chunk_all(method: "str | ChunkingMethod" = ChunkingMethod.RECURSIVE):
    """Chunkt ALLE .md-Dateien aus data/processed mit der gewählten Methode.

    Speichert NICHT in die DB – gibt nur eine Übersicht aus (Chunk-Anzahl, Seiten-
    spanne je Datei) und liefert eine Liste (dateiname, anzahl_chunks) zurück.
    """
    method = ChunkingMethod.from_value(method)
    md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        print(f"Keine Markdown-Dateien in {PROCESSED_DIR}")
        return []

    print(f"Chunking-Methode: {method.value} | {len(md_files)} Datei(en)\n")
    results: list[tuple[str, int]] = []
    for md in md_files:
        try:
            records = chunk_file(str(md), method)
            pages = sorted(
                {r.metadata.get("page") for r in records if r.metadata.get("page") is not None}
            )
            span = f"S. {pages[0]}-{pages[-1]}" if pages else "keine Seiten"
            print(f"  {md.name}: {len(records)} Chunks ({span})")
            results.append((md.name, len(records)))
        except Exception as exc:
            print(f"  [FEHLER] {md.name}: {exc}")

    total = sum(n for _, n in results)
    print(f"\nFertig: {len(results)} Datei(en), {total} Chunks insgesamt.")
    return results


if __name__ == "__main__":
    import sys

    # Optionales Methoden-Argument, z. B.:  python chunking.py markdown
    chosen = sys.argv[1] if len(sys.argv) > 1 else ChunkingMethod.SEMANTIC
    chunk_all(chosen)
