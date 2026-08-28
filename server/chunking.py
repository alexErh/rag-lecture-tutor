"""Chunking-Logik: Markdown-Text in (formelschonende) Chunks zerlegen.

Zwei Chunking-Methoden, unterscheidbar über das Enum ChunkingMethod:

* RECURSIVE  – zeichenbasiertes Splitten mit RecursiveCharacterTextSplitter
               (kleine, gleichmäßige Chunks; ignoriert die Dokumentstruktur).
* MARKDOWN   – strukturbasiertes Splitten mit MarkdownHeaderTextSplitter: teilt
               zuerst an den Überschriften (#/##/###), sodass ein Chunk nie über
               Abschnittsgrenzen läuft, und hängt die Überschriften-Hierarchie als
               Metadaten (h1/h2/h3) an. Große Abschnitte werden anschließend auf
               eine embeddbare Größe begrenzt.

Beide Methoden schützen LaTeX-Formeln vor dem Zerschneiden (siehe Formel-Schutz).
Jeder Chunk wird mit metadata['method'] markiert, damit die DB gezielt nach den
Chunks einer bestimmten Methode suchen kann.
"""

import re
from enum import Enum

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
    SEMANTIC = "semantic"

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


# ========= SPLITTER PRO METHODE =========

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


def chunk_file(path: str, method: "str | ChunkingMethod" = ChunkingMethod.RECURSIVE):
    """Liest eine Markdown-Datei ein und zerlegt sie formelschonend in Chunks.

    Args:
        path: Pfad zur Markdown-Datei.
        method: Chunking-Methode (ChunkingMethod oder deren String-Wert).

    Returns:
        Liste von Document-Chunks; jeder trägt metadata['source'] und
        metadata['method'].
    """
    method = ChunkingMethod.from_value(method)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    all_chunks: list[Document] = []
    # Pro Seite getrennt chunken -> jeder Chunk trägt seine Seitenzahl und läuft
    # nie über eine Seitengrenze.
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
        for chunk in chunks:
            chunk.page_content = _unmask_math(chunk.page_content, formulas)
            chunk.metadata["source"] = path
            chunk.metadata["method"] = method.value
            if page_no is not None:
                chunk.metadata["page"] = page_no

        all_chunks.extend(chunks)

    return all_chunks
