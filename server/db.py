import chromadb
import re
import uuid
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ========= CHROMA =========

# Absoluter Pfad (relativ zu dieser Datei), damit Server und Notebook-Ingest
# dieselbe DB nutzen – unabhängig vom aktuellen Arbeitsverzeichnis.
_BASE_DIR = Path(__file__).resolve().parent
client = chromadb.PersistentClient(path=str(_BASE_DIR / "VectorDB"))

collection = client.get_or_create_collection(
    name="VectorDB",
)


# ========= CHUNKER =========

simple_chunker = RecursiveCharacterTextSplitter(
    chunk_size=100,
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


# ========= FUNCTIONS =========

def chunk_file(path: str):
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

# Lese Path von Markdown ein um es Chunken zu lassen und in die DB zu speichern
def add_document(path: str):
    chunks = chunk_file(path)

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

# Bekommt den Prompt als Query und gibt die passenden Chunks zurück
def retrieve_chunks(query: str, n: int = 3):
    # eventuell retreival anpassen/verbessern
    return collection.query(
        query_texts=[query],
        n_results=n
    )


# ========= API =========

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    n: int = 3


@app.post("/query")
def query(request: QueryRequest):
    results = retrieve_chunks(
        request.query,
        request.n
    )

    return {
        "documents": results["documents"],
        "distances": results["distances"],
        "metadatas": results["metadatas"],
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "db:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )