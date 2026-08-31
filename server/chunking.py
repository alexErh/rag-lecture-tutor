"""Chunking-Logik: Markdown-Text in (formelschonende) Chunks zerlegen.

Chunking-Methoden, unterscheidbar über das Enum ChunkingMethod:

* RECURSIVE  – zeichenbasiertes Splitten mit RecursiveCharacterTextSplitter
               (kleine, gleichmäßige Chunks; ignoriert die Dokumentstruktur).
* MARKDOWN   – strukturbasiertes Splitten mit MarkdownHeaderTextSplitter: teilt
               zuerst an den Überschriften (#/##/###) und hängt die Hierarchie als
               Metadaten (h1/h2/h3) an. Große Abschnitte werden begrenzt.
* LATE       – Late Chunking mit Chonkie: der (lange) Text wird ZUERST embeddet
               (Token-Level), dann werden Chunk-Grenzen gelegt und die Token-Vektoren
               PRO Chunk gemittelt. Jeder Chunk-Vektor trägt so Dokumentkontext.
               Anders als die anderen Methoden liefert LATE bereits die Embeddings
               mit (ChunkRecord.embedding); diese werden in einer eigenen Collection
               gespeichert (siehe database.py), weil ihre Dimension vom Standard-
               Embedding-Modell abweicht.

Die (text-basierten) Methoden schützen LaTeX-Formeln vor dem Zerschneiden (siehe
Formel-Schutz). Jeder Chunk wird mit metadata['method'] markiert, damit die DB
gezielt nach den Chunks einer bestimmten Methode suchen kann.

Perspektive: die übrigen Methoden werden schrittweise ebenfalls auf Chonkie
umgestellt (TokenChunker = fixed-size, SemanticChunker, RecursiveChunker/Struktur);
das gemeinsame Rückgabeformat ChunkRecord ist dafür schon vorbereitet.
"""

import server.hf_offline  # noqa: F401 -- MUSS zuerst stehen: HF-Offline vor HF-nutzenden Imports

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from chonkie import SemanticChunker


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

# Embedding-Modell (paraphrase-multilingual-MiniLM-L12-v2) verarbeitet max. 128
# Tokens (~400-500 Zeichen). Größere Chunks würden beim Embedden abgeschnitten,
# darum begrenzen beide Methoden die Chunk-Größe entsprechend.

# RECURSIVE: kleine, gleichmäßige Chunks rein nach Trennzeichen.
_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " "],
)

# MARKDOWN: erst an Überschriften trennen (Struktur + Header-Metadaten) ...
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_md_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_MD_HEADERS,
    strip_headers=False,  # Überschrift im Chunk-Text belassen (guter Kontext)
)
# ... dann zu große Abschnitte auf embeddbare Größe begrenzen (Metadaten bleiben).
_md_size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " "],
)

_semantic_splitter = SemanticChunker(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    threshold=0.7,
    chunk_size=4096,
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

# Platzhalter aus Zeichen, die NICHT in den Splitter-Separatoren vorkommen
# (kein Leerzeichen, kein Zeilenumbruch, kein ". "). Ohne diese Trennzeichen kann
# der RecursiveCharacterTextSplitter den Platzhalter nicht in der Mitte trennen –
# im Extremfall entsteht ein etwas zu großer Chunk, die Formel bleibt aber ganz.
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
# lang genug sein, damit der Kontext etwas bringt; das Standard-128-Token-Modell ist
# dafür zu kurz. Modell (und Chunk-Größe) sind per .env konfigurierbar.
#
# WICHTIG (deutschsprachiges Material): ein MEHRSPRACHIGES Long-Context-Modell wählen,
# z. B. LATE_EMBEDDING_MODEL=intfloat/multilingual-e5-base oder jinaai/jina-embeddings-v3.
LATE_EMBEDDING_MODEL = os.getenv("LATE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
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


def _late_chunks(text: str, path: str) -> "list[ChunkRecord]":
    """Late Chunking über das ganze Dokument; Seite via Zeichen-Offset zugeordnet."""
    _, chunker = _get_late()
    clean_text, boundaries = _strip_page_markers(text)

    records: list[ChunkRecord] = []
    for chunk in chunker.chunk(clean_text):
        metadata = {"source": path, "method": ChunkingMethod.LATE.value}
        page = _page_at(chunk.start_index, boundaries)
        if page is not None:
            metadata["page"] = page
        vector = chunk.embedding
        embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        records.append(ChunkRecord(text=chunk.text, metadata=metadata, embedding=embedding))
    return records


# ========= SPLITTER PRO METHODE (text-basiert) =========

def _recursive_chunks(masked_text: str, path: str) -> list[Document]:
    document = Document(page_content=masked_text, metadata={"source": path})
    return _recursive_splitter.split_documents([document])


def _markdown_chunks(masked_text: str, path: str) -> list[Document]:
    # 1) An Überschriften trennen -> Documents mit h1/h2/h3 in den Metadaten.
    sections = _md_header_splitter.split_text(masked_text)
    # 2) Zu große Abschnitte auf embeddbare Größe begrenzen (Header-Metadaten bleiben).
    return _md_size_splitter.split_documents(sections)

# Chonkie Semantic Chunking
def _semantic_chunks(masked_text: str, path: str) -> list[Document]:
    chunks = _semantic_splitter.chunk(masked_text)
    # Wandelt einen Chonkie Chunk in ein Dokument um. Wichtig für die Weiterverarbeitung, da sie auf Documents basiert.
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "source": path,
            },
        )
        for chunk in chunks
    ]


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
                chunks = _markdown_chunks(masked_text, path)
            case ChunkingMethod.RECURSIVE:
                chunks = _recursive_chunks(masked_text, path)
            case ChunkingMethod.SEMANTIC:
                chunks = _semantic_chunks(masked_text, path)
        # ... Formeln je Chunk wiederherstellen und Metadaten vereinheitlichen.
        filename = Path(path).stem
        with open('data/chunks/' + filename + f'_{method.value}_chunks.txt', 'w', encoding="utf-8") as f:
            for i,chunk in enumerate(chunks):
                content = _unmask_math(chunk.page_content, formulas)
                metadata = dict(chunk.metadata)  # enthält bei MARKDOWN h1/h2/h3
                metadata["source"] = path
                metadata["method"] = method.value
                if page_no is not None:
                    metadata["page"] = page_no
                records.append(ChunkRecord(text=content, metadata=metadata))


                f.write(f'========== CHUNK {i} ========== \n')
                f.write(chunk.page_content + '\n')


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
    chosen = sys.argv[1] if len(sys.argv) > 1 else ChunkingMethod.MARKDOWN
    chunk_all(chosen)



