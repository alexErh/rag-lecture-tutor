import chromadb
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import uvicorn

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# PDF -> Markdown-Konvertierung (docling)
from pdf_to_markdown import convert_pdf, _build_converter, RAW_DIR

# .env laden (LOCAL, LOCAL_BASE_URL, LOCAL_MODEL_NAME, NVIDIA_API_KEY)
load_dotenv()


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


# ========= AGENT =========
# Der Agent wird beim Import (und damit beim API-Start) aufgebaut. Er nutzt ein
# Retrieval-Tool, das DIREKT in-process auf die DB zugreift (kein HTTP-Umweg).

SYSTEM_PROMPT = (
    "Du bist ein Tutor für Vorlesungsinhalte. Beantworte Fragen nur auf Basis des "
    "bereitgestellten Kontexts. Erkläre klar, korrekt und verständlich. Wenn "
    "Informationen fehlen oder unsicher sind, sage das ausdrücklich. Erfinde nichts "
    "und spekuliere nicht. Nutze Fachbegriffe korrekt und erkläre sie kurz, wenn nötig. "
    "Gib, falls Informationen aus der Funktion search_lecture_docs entnommen werden – das "
    "heißt, dass die Informationen aus einer Datei kommen –, immer den Dateipfad in "
    "folgendem Format an: *Quelle*: `quelle`. Beispiel: *Quelle*: `data/raw/somefile.txt`"
)

# LOCAL=true -> lokales Ollama-Modell; LOCAL=false -> Nvidia NIM API.
LOCAL = os.getenv("LOCAL", "true").strip().lower() in ("1", "true", "yes", "ja")


@tool(
    "search_lecture_docs",
    description=(
        "Searches the lecture documents using semantic similarity and retrieves the "
        "most relevant text chunks for answering the user's question. Use this tool when "
        "the user asks a question that requires information from the lecture documents. "
        "The `query` parameter must contain a concise semantic search query representing "
        "the user's information need. Query rules: preserve important technical terms, "
        "concepts, names and keywords; do not include instructions to the assistant, "
        "unnecessary conversational text or the user's entire conversation; formulate the "
        "query so that it is useful for semantic vector search; for conceptual questions "
        "keep the important concepts from the original question. Examples: "
        "User: 'Was ist Retrieval Augmented Generation?' -> query: 'Retrieval Augmented "
        "Generation'. User: 'Wie funktioniert der RecursiveCharacterTextSplitter?' -> "
        "query: 'RecursiveCharacterTextSplitter Funktionsweise'. If the user explicitly "
        "asks about a specific file, DO NOT use this tool to retrieve all chunks of that "
        "file; use the get_file_info tool instead. This tool performs semantic similarity "
        "search; it does not search metadata or retrieve all chunks belonging to a file."
    ),
)
def search_lecture_docs(query: str):
    """Holt die passenden Chunks per semantischer Suche aus der Vektor-DB."""
    results = retrieve_chunks(query, 3)
    return {
        "documents": results["documents"],
        "distances": results["distances"],
        "metadatas": results["metadatas"],
    }


@tool(
    "get_file_info",
    description=(
        "Retrieves all information/chunks from a specific lecture document. IMPORTANT: "
        "the `filename` parameter MUST contain ONLY the filename explicitly mentioned by "
        "the user, including its file extension, never the user's complete question, "
        "extra words (such as 'Was steht in', 'Datei', 'Inhalt von') or a file path. "
        "Examples: 'Was steht in example.md?' -> filename: 'example.md'. "
        "'Erkläre mir den Inhalt von lecture_03.pdf.' -> filename: 'lecture_03.pdf'. "
        "Use this tool ONLY when the user explicitly identifies a specific file; if no "
        "specific filename is mentioned, do not use this tool."
    ),
)
def get_file_info(filename: str):
    """Holt alle Chunks einer bestimmten Datei über den Metadaten-Filter."""
    results = retrieve_through_metadata(filename)
    return {
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    }


def _build_model():
    if LOCAL:
        return ChatOllama(
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LOCAL_MODEL_NAME", "llama3.2:3b"),
        )
    return ChatOpenAI(
        model="meta/llama-3.1-8b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )


agent = create_agent(
    _build_model(),
    tools=[search_lecture_docs, get_file_info],
    system_prompt=SYSTEM_PROMPT,
)


def ask_agent(query: str) -> str:
    """Stellt dem Agenten eine Frage und gibt die Antwort als Text zurück."""
    result = agent.invoke({"messages": [("user", query)]})
    return result["messages"][-1].content


# ========= API =========

def _ollama_keep_alive(keep_alive, timeout: int = 300) -> str:
    """Sendet eine Leer-Anfrage an Ollama, um das Modell zu laden bzw. zu entladen.

    keep_alive=-1 lädt das Modell und hält es im Speicher; keep_alive=0 entlädt es
    sofort. Gibt den Modellnamen zurück.
    """
    import json
    import urllib.request

    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("LOCAL_MODEL_NAME", "llama3.2:3b")
    payload = json.dumps(
        {"model": model_name, "prompt": "", "keep_alive": keep_alive}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        resp.read()
    return model_name


def _warmup_ollama() -> None:
    """Lädt das lokale Ollama-Modell beim Start in den Speicher (keep_alive=-1).

    So ist das Modell schon vor der ersten /ask-Anfrage bereit. Nur relevant im
    lokalen Modus (LOCAL=True); scheitert leise, falls Ollama (noch) nicht läuft.
    """
    if not LOCAL:
        return
    try:
        print("[startup] Lade Ollama-Modell vor ...")
        model_name = _ollama_keep_alive(-1, timeout=300)
        print(f"[startup] Ollama-Modell '{model_name}' geladen und bereit.")
    except Exception as exc:
        print(f"[startup] Warmup übersprungen (läuft Ollama?): {exc}")


def _unload_ollama() -> None:
    """Entlädt das lokale Ollama-Modell beim Herunterfahren aus dem Speicher.

    Gibt den RAM wieder frei (keep_alive=0). Nur im lokalen Modus; scheitert leise.
    """
    if not LOCAL:
        return
    try:
        print("[shutdown] Entlade Ollama-Modell ...")
        model_name = _ollama_keep_alive(0, timeout=30)
        print(f"[shutdown] Ollama-Modell '{model_name}' entladen.")
    except Exception as exc:
        print(f"[shutdown] Entladen übersprungen: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Beim API-Start: Modell vorladen ...
    _warmup_ollama()
    yield
    # ... beim Herunterfahren: Modell wieder entladen.
    _unload_ollama()


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    n: int = 3

class FileRequest(BaseModel):
    filename: str


class AskRequest(BaseModel):
    query: str


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


@app.post("/document")
def get_filechunks(request: FileRequest):
    results = retrieve_through_metadata(request.filename)
    return {
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    }


# Stellt dem Backend-Agenten eine Frage und gibt die generierte Antwort zurück.
@app.post("/ask")
def ask(request: AskRequest):
    answer = ask_agent(request.query)
    return {"answer": answer}


# Nimmt hochgeladene PDF-Dateien entgegen, wandelt sie mit docling in Markdown um
# und schreibt die (formelschonend) gechunkten Abschnitte in die Vektor-DB.
@app.post("/ingest")
def ingest(
    files: list[UploadFile] = File(...),
    formulas: bool = Form(False),      # Formeln -> LaTeX (langsam, CPU)
    ocr: bool = Form(False),           # OCR (nur für gescannte PDFs nötig, langsam)
    delete_pdfs: bool = Form(True),    # verarbeitete PDFs nach dem Ingest aus raw/ löschen
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
            n_chunks = add_document(str(md_path))
            results.append({
                "file": filename,
                "markdown": md_path.name,
                "chunks": n_chunks,
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
    # Debug toggle. True = Normales Starten des Servers, False = Beliebige Funktionen ausführen
    if(True):
        import uvicorn

        uvicorn.run(
            "db:app",
            host="127.0.0.1",
            port=8000,
            reload=True
        )
    else:

        add_document("data/processed/example.md")