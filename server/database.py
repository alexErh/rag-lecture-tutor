"""DB-Operationen: Zugriff auf die ChromaDB (Speichern & Abrufen von Chunks).

Kapselt den ChromaDB-Client, die Collection und das Embedding-Modell sowie die
Funktionen zum Hinzufügen und Abrufen von Chunks. Das Chunking selbst liegt in
chunking.py.
"""

import hf_offline  # noqa: F401 -- MUSS zuerst stehen: HF-Offline vor dem Embedding-Modell

import hashlib
import time
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunking import (
    chunk_file,
    embed_query_late,
    ChunkingMethod,
    LATE_EMBEDDING_MODEL,
    PROCESSED_DIR,
)


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


# ========= LATE-CHUNKING-COLLECTION =========
# Late Chunks bringen ihre Embeddings selbst mit (Chonkie, Long-Context-Modell). Deren
# Vektor-Dimension weicht vom Standard-Modell ab -> eigene Collection OHNE
# embedding_function (wir liefern immer vorab berechnete Vektoren bzw. Query-Vektoren).
# Name je Modell-Hash, damit ein Modellwechsel nicht mit alten Dimensionen kollidiert.
_late_collection_cache = None


def _late_collection():
    global _late_collection_cache
    if _late_collection_cache is None:
        model_hash = hashlib.md5(LATE_EMBEDDING_MODEL.encode()).hexdigest()[:12]
        _late_collection_cache = client.get_or_create_collection(
            name=f"late_{model_hash}",
            metadata={"hnsw:space": "cosine"},
        )
    return _late_collection_cache


# ========= OPERATIONEN =========

# Lese Path von Markdown ein um es Chunken zu lassen und in die DB zu speichern
def add_document(path: str, method: "str | ChunkingMethod" = ChunkingMethod.RECURSIVE):
    """Chunkt eine Markdown-Datei mit der gewählten Methode und speichert sie.

    Die Methoden koexistieren pro Datei: ein erneuter Ingest ersetzt nur die Chunks
    DERSELBEN Methode (idempotent pro Datei+Methode). Text-basierte Methoden landen in
    der Standard-Collection (die den Text selbst embeddet); LATE bringt fertige
    Embeddings mit und landet in der eigenen Late-Collection.
    """
    method = ChunkingMethod.from_value(method)
    records = chunk_file(path, method)

    is_late = method is ChunkingMethod.LATE
    target = _late_collection() if is_late else collection

    # Nur Chunks derselben Quelle UND Methode entfernen (keine Duplikate).
    try:
        target.delete(where={"$and": [{"source": path}, {"method": method.value}]})
    except Exception:
        pass

    if not records:
        return 0

    ids = [str(uuid.uuid4()) for _ in records]
    documents = [r.text for r in records]
    metadatas = [r.metadata for r in records]

    if is_late:
        # Vorab berechnete Vektoren mitgeben -> Chroma embeddet NICHT selbst.
        target.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=[r.embedding for r in records],
        )
    else:
        # Text-basiert -> die embedding_function der Collection embeddet den Text.
        target.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(records)


def _method_filter(method: "str | ChunkingMethod | None"):
    """Chroma-where-Filter für die Methode (None -> kein Filter, sucht in allen)."""
    if method is None:
        return None
    return {"method": ChunkingMethod.from_value(method).value}


def _is_late(method: "str | ChunkingMethod | None") -> bool:
    return method is not None and ChunkingMethod.from_value(method) is ChunkingMethod.LATE


# Bekommt den Prompt als Query und gibt die passenden Chunks zurück
def retrieve_chunks(query: str, n: int = 3, method: "str | ChunkingMethod | None" = None):
    # LATE: eigene Collection, Query mit demselben Late-Modell embedden (gleicher Raum).
    print("Retrieving chunks...\n", "Chunking Method:\t", method)
    if _is_late(method):
        return _late_collection().query(
            query_embeddings=[embed_query_late(query)],
            n_results=n,
        )
    # Sonst: Standard-Collection; method=None -> alle, sonst nur die gewählte Methode.
    return collection.query(
        query_texts=[query],
        n_results=n,
        where=_method_filter(method),
    )


def retrieve_through_metadata(filename: str, method: "str | ChunkingMethod | None" = None):
    if _is_late(method):
        target, where = _late_collection(), None  # Late-Collection enthält nur Late-Chunks
    else:
        target, where = collection, _method_filter(method)

    results = target.get(include=["documents", "metadatas"], where=where)

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


# ========= CLI: alle Markdown-Dateien aus data/processed in die DB laden =========

def add_all(method: "str | ChunkingMethod" = ChunkingMethod.RECURSIVE):
    """Chunkt ALLE .md aus data/processed mit der gewählten Methode und speichert sie
    in die DB (via add_document). Danach sind sie über /query und /ask (mit passendem
    method) abfragbar. Idempotent pro Datei+Methode.
    """
    method = ChunkingMethod.from_value(method)
    md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        print(f"Keine Markdown-Dateien in {PROCESSED_DIR}")
        return []

    print(f"In DB laden | Methode: {method.value} | {len(md_files)} Datei(en)\n")
    results: list[tuple[str, int]] = []
    total_time = 0.0
    for md in md_files:
        try:
            t0 = time.perf_counter()
            n = add_document(str(md), method=method)
            dt = time.perf_counter() - t0
            total_time += dt
            print(f"  {md.name}: {n} Chunks gespeichert | {dt:.2f}s")
            results.append((md.name, n))
        except Exception as exc:
            print(f"  [FEHLER] {md.name}: {exc}")

    total = sum(n for _, n in results)
    target = _late_collection() if method is ChunkingMethod.LATE else collection
    print(
        f"\nFertig: {len(results)} Datei(en), {total} Chunks gespeichert. "
        f"Zeit: {total_time:.2f}s (Methode: {method.value}). "
        f"Collection-Gesamt: {target.count()}"
    )
    return results


if __name__ == "__main__":
    # add_all einmal pro Chunking-Methode aufrufen (insgesamt vier Läufe:
    # recursive, markdown, semantic, late). Alle Methoden koexistieren in der DB.
    for method in ChunkingMethod:
        print(f"\n{'=' * 70}\nMethode: {method.value}\n{'=' * 70}")
        add_all(method)
