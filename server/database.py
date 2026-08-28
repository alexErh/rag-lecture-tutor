"""DB-Operationen: Zugriff auf die ChromaDB (Speichern & Abrufen von Chunks).

Kapselt den ChromaDB-Client, die Collection und das Embedding-Modell sowie die
Funktionen zum Hinzufügen und Abrufen von Chunks. Das Chunking selbst liegt in
chunking.py.
"""

import hf_offline  # noqa: F401 -- MUSS zuerst stehen: HF-Offline vor dem Embedding-Modell

import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_file, ChunkingMethod


# ========= CHROMA =========

# Absoluter Pfad (relativ zu dieser Datei), damit Server und Notebook-Ingest
# dieselbe DB nutzen – unabhängig vom aktuellen Arbeitsverzeichnis.
_BASE_DIR = Path(__file__).resolve().parent
client = chromadb.PersistentClient(path=str(_BASE_DIR / "VectorDB"))

# Mehrsprachiges Embedding-Modell – passend für deutschsprachige Vorlesungen
# (Chromas Default all-MiniLM-L6-v2 ist englischlastig). Läuft lokal; das Modell
# wird beim ersten Aufruf einmalig heruntergeladen.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="VectorDB",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # Cosine passt zu SentenceTransformer-Vektoren
)


# ========= OPERATIONEN =========

# Lese Path von Markdown ein um es Chunken zu lassen und in die DB zu speichern
def add_document(path: str, method: "str | ChunkingMethod" = ChunkingMethod.RECURSIVE):
    """Chunkt eine Markdown-Datei mit der gewählten Methode und speichert sie.

    Die beiden Methoden koexistieren pro Datei: ein erneuter Ingest ersetzt nur die
    Chunks DERSELBEN Methode (idempotent pro Datei+Methode), lässt die Chunks der
    jeweils anderen Methode aber unberührt.
    """
    method = ChunkingMethod.from_value(method)
    chunks = chunk_file(path, method)

    # Nur Chunks derselben Quelle UND Methode entfernen (keine Duplikate).
    try:
        collection.delete(where={"$and": [{"source": path}, {"method": method.value}]})
    except Exception as e:
        print(f'EXCEPTION add_document: {e}')

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return len(chunks)


def _method_filter(method: "str | ChunkingMethod | None"):
    """Chroma-where-Filter für die Methode (None -> kein Filter, sucht in allen)."""
    if method is None:
        return None
    return {"method": ChunkingMethod.from_value(method).value}


# Bekommt den Prompt als Query und gibt die passenden Chunks zurück
def retrieve_chunks(query: str, n: int = 3, method: "str | ChunkingMethod | None" = None):
    # method=None -> alle Chunks; sonst nur die der gewählten Methode.
    return collection.query(
        query_texts=[query],
        n_results=n,
        where=_method_filter(method),
    )


def retrieve_through_metadata(filename: str, method: "str | ChunkingMethod | None" = None):
    results = collection.get(
        include=["documents", "metadatas"],
        where=_method_filter(method),
    )

    filtered = [
        (document, metadata)
        for document, metadata in zip(
            results["documents"],
            results["metadatas"]
        )
        if metadata.get("source", "").endswith(filename)
    ]

    return {
        "documents": [document for document, _ in filtered],
        "metadatas": [metadata for _, metadata in filtered],
    }