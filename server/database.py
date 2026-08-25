"""DB-Operationen: Zugriff auf die ChromaDB (Speichern & Abrufen von Chunks).

Kapselt den ChromaDB-Client, die Collection und das Embedding-Modell sowie die
Funktionen zum Hinzufügen und Abrufen von Chunks. Das Chunking selbst liegt in
chunking.py.
"""

import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_file


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
def add_document(path: str):
    chunks = chunk_file(path)

    # Vorhandene Chunks derselben Quelle entfernen -> erneuter Ingest erzeugt
    # keine Duplikate (idempotent pro Datei).
    try:
        collection.delete(where={"source": path})
    except Exception:
        pass

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return len(chunks)


# Bekommt den Prompt als Query und gibt die passenden Chunks zurück
def retrieve_chunks(query: str, n: int = 3):
    # eventuell retreival anpassen/verbessern
    return collection.query(
        query_texts=[query],
        n_results=n
    )


def retrieve_through_metadata(filename: str):
    results = collection.get(
        include=["documents", "metadatas"]
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
