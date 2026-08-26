"""API-Code: FastAPI-App, Endpunkte und Server-Start.

Starten mit:  uvicorn api:app --reload   (oder:  python api.py)

Baut auf:
  - agent.py            -> RAG-Agent (/ask)
  - database.py         -> DB-Operationen (ChromaDB)
  - chunking.py         -> Chunking-Logik (indirekt über database.py)
  - pdf_to_markdown.py  -> PDF -> Markdown (docling)
  - ollama_lifecycle.py -> lokales Modell beim Start/Stop laden/entladen
"""
from typing import List
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

# PDF -> Markdown-Konvertierung (docling)
from pdf_to_markdown import convert_pdf, _build_converter, RAW_DIR

# Chunking-Methoden (Enum, geteilt mit database.py/agent.py)
from chunking import ChunkingMethod

# DB-Operationen
from database import (
    collection,
    add_document,
    retrieve_chunks,
    retrieve_through_metadata,
)

# Agent (Fragen beantworten)
from agent import ask_agent

# Ollama-Lebenszyklus (optional: Modul darf fehlen -> No-op-Fallback)
try:
    from ollama_lifecycle import warmup_ollama, unload_ollama
except ImportError:
    def warmup_ollama() -> None:
        pass

    def unload_ollama() -> None:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Beim API-Start: Modell vorladen ...
    warmup_ollama()
    yield
    # ... beim Herunterfahren: Modell wieder entladen.
    unload_ollama()


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    n: int = 3
    # None -> methodenübergreifend suchen; sonst nur Chunks dieser Methode.
    method: ChunkingMethod | None = None

class FileRequest(BaseModel):
    filename: str
    method: ChunkingMethod | None = None


class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    messages: List[Message]
    method: ChunkingMethod | None = None

@app.post("/query")
def query(request: QueryRequest):
    results = retrieve_chunks(
        request.query,
        request.n,
        method=request.method,
    )
    return {
        "documents": results["documents"],
        "distances": results["distances"],
        "metadatas": results["metadatas"],
    }


@app.post("/document")
def get_filechunks(request: FileRequest):
    results = retrieve_through_metadata(request.filename, method=request.method)
    return {
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    }


# Stellt dem Backend-Agenten eine Frage und gibt die generierte Antwort zurück.
@app.post("/ask")
def ask(request: AskRequest):
    messages = [
        (msg.role, msg.content)
        for msg in request.messages
    ]
    answer = ask_agent(messages, method=request.method)
    return {"answer": answer}


# Nimmt hochgeladene PDF-Dateien entgegen, wandelt sie mit docling in Markdown um
# und schreibt die (formelschonend) gechunkten Abschnitte in die Vektor-DB.
@app.post("/ingest")
def ingest(
    files: list[UploadFile] = File(...),
    formulas: bool = Form(False),      # Formeln -> LaTeX (langsam, CPU)
    ocr: bool = Form(False),           # OCR (nur für gescannte PDFs nötig, langsam)
    delete_pdfs: bool = Form(True),    # verarbeitete PDFs nach dem Ingest aus raw/ löschen
    method: ChunkingMethod = Form(ChunkingMethod.RECURSIVE),  # Chunking-Methode
):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # docling-Converter einmal pro Request aufbauen und wiederverwenden.
    # Standard: schnell (OCR & Formel-Anreicherung aus).
    converter = _build_converter(enable_formulas=formulas, enable_ocr=ocr)

    results = []

    # --- Phase 1: ALLE hochgeladenen PDFs im raw-Ordner speichern ---
    saved = []  # (originaler_dateiname, Pfad_im_raw_Ordner)
    for f in files:
        filename = f.filename or ""
        if not filename.lower().endswith(".pdf"):
            results.append({"file": filename, "status": "übersprungen (keine PDF)"})
            continue
        dest = RAW_DIR / Path(filename).name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append((filename, dest))

    # --- Phase 2: alle gespeicherten PDFs -> Markdown -> Chunks -> DB ---
    processed = []  # raw-PDFs, die erfolgreich verarbeitet wurden (zum Löschen)
    for filename, pdf_path in saved:
        try:
            md_path = convert_pdf(pdf_path, converter=converter, overwrite=True)
            n_chunks = add_document(str(md_path), method=method)
            results.append({
                "file": filename,
                "markdown": md_path.name,
                "chunks": n_chunks,
                "method": method.value,
                "status": "ok",
            })
            processed.append(pdf_path)
        except Exception as exc:
            results.append({"file": filename, "status": f"Fehler: {exc}"})

    # --- Phase 3: verarbeitete PDFs aus dem raw-Ordner löschen (optional) ---
    deleted = 0
    if delete_pdfs:
        for pdf_path in processed:
            try:
                pdf_path.unlink()
                deleted += 1
            except Exception:
                pass

    return {
        "results": results,
        "deleted_pdfs": deleted,
        "collection_count": collection.count(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
