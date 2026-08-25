"""Chunking-Logik: Markdown-Text in (formelschonende) Chunks zerlegen.

Enthält den Text-Splitter und den Formel-Schutz, damit LaTeX-Formeln beim
Splitten nicht zerschnitten werden.
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ========= CHUNKER =========

simple_chunker = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " "]
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


def chunk_file(path: str):
    """Liest eine Markdown-Datei ein und zerlegt sie formelschonend in Chunks."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Formeln vor dem Splitten schützen ...
    masked_text, formulas = _mask_math(text)

    document = Document(
        page_content=masked_text,
        metadata={"source": path}
    )

    chunks = simple_chunker.split_documents([document])

    # ... und in jedem Chunk wiederherstellen.
    for chunk in chunks:
        chunk.page_content = _unmask_math(chunk.page_content, formulas)

    return chunks
