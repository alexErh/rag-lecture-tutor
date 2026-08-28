"""Agent-Code: Retrieval-Tools, Modellwahl (lokal/Cloud) und der RAG-Agent.

Der Agent wird beim Import (und damit beim API-Start) aufgebaut. Er nutzt
Retrieval-Tools, die DIREKT in-process auf die DB zugreifen (kein HTTP-Umweg).
"""

import hf_offline  # noqa: F401 -- MUSS zuerst stehen: HF-Offline vor HF-nutzenden Imports

import contextvars
import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from chunking import ChunkingMethod
from database import retrieve_chunks, retrieve_through_metadata

# .env laden (LOCAL, LOCAL_BASE_URL, LOCAL_MODEL_NAME, NVIDIA_API_KEY)
load_dotenv()

# Aktuelle Chunking-Methode des laufenden /ask-Requests. Wird von ask_agent gesetzt
# und von den Retrieval-Tools gelesen, damit der Agent gezielt in den Chunks EINER
# Methode sucht (None -> methodenübergreifend). Über eine ContextVar, weil die Tools
# innerhalb desselben synchronen Aufrufs laufen und keinen Parameter durchgereicht
# bekommen.
_request_method: contextvars.ContextVar = contextvars.ContextVar(
    "request_method", default=None
)


SYSTEM_PROMPT = (
    "Du bist ein Tutor für Vorlesungsinhalte. Beantworte Fragen nur auf Basis des "
    "bereitgestellten Kontexts. Erkläre klar, korrekt und verständlich. Wenn "
    "Informationen fehlen oder unsicher sind, sage das ausdrücklich. Erfinde nichts "
    "und spekuliere nicht. Nutze Fachbegriffe korrekt und erkläre sie kurz, wenn nötig. Ausgaben sollen im Markdown-Format "
    "sein, sodass man es in einem Markdown-Reader anzeigen könnte. Formelblöcke also mit $$ darstellen."
    "Gib, wann immer du Informationen aus den Funktionen search_lecture_docs oder "
    "get_file_info verwendest (die Informationen stammen dann aus einer Datei), immer die "
    "Quelle an – mit dem Dateipfad (Feld 'source') UND, sofern das Feld 'page' des "
    "verwendeten Chunks vorhanden (nicht null) ist, der Seitenzahl. Format MIT Seite: "
    "*Quelle*: `quelle`, S. `page`. Beispiel: *Quelle*: `data/processed/08_LinAbb.md`, S. 7. "
    "Fehlt 'page' (null), nur den Pfad angeben: *Quelle*: `quelle`. Nutzt du mehrere Chunks "
    "aus verschiedenen Quellen oder von verschiedenen Seiten, nenne alle jeweiligen Quellen."
)

# LOCAL=true -> lokales Ollama-Modell; LOCAL=false -> Nvidia NIM API.
LOCAL = os.getenv("LOCAL", "true").strip().lower() in ("1", "true", "yes", "ja")


def _format_hits(documents, metadatas, distances=None):
    """Bereitet Chunks flach auf: pro Treffer content + source + page (+ distance).

    Macht Quelle und Seite für das Modell explizit sichtbar, damit es korrekt
    zitieren kann, statt sie aus verschachtelten Metadaten ziehen zu müssen.
    """
    hits = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        hit = {
            "content": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),  # None, falls (noch) nicht getaggt
        }
        if distances is not None:
            hit["distance"] = distances[i]
        hits.append(hit)
    return hits


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
    results = retrieve_chunks(query, 5, method=_request_method.get())
    # collection.query liefert je Abfrage verschachtelte Listen -> [0].
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []
    return {"results": _format_hits(documents, metadatas, distances)}


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
    results = retrieve_through_metadata(filename, method=_request_method.get())
    return {"results": _format_hits(results["documents"], results["metadatas"])}


def _build_model():
    if LOCAL:
        return ChatOllama(
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LOCAL_MODEL_NAME", "llama3.2:3b"),
        )
    return ChatOpenAI(
        model= "openai/gpt-oss-20b", #"meta/llama-3.1-8b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )


agent = create_agent(
    _build_model(),
    tools=[search_lecture_docs, get_file_info],
    system_prompt=SYSTEM_PROMPT,
)


def _has_source_citation(answer: str) -> bool:
    """Grobe Prüfung, ob der Agent die Quelle bereits selbst angegeben hat."""
    return "quelle" in answer.lower()


def _collect_retrieved_sources(messages) -> "dict[str, set]":
    """Sammelt aus den Tool-Ergebnissen: Dateiname -> Menge der Seitenzahlen.

    Reihenfolge des ersten Auftretens bleibt erhalten (dict ist geordnet).
    """
    sources: dict[str, set] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        payload = msg.content
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                continue
        if not isinstance(payload, dict):
            continue
        for hit in payload.get("results", []):
            src = hit.get("source")
            if not src:
                continue
            name = os.path.basename(str(src).replace("\\", "/"))
            pages = sources.setdefault(name, set())
            if hit.get("page") is not None:
                pages.add(hit["page"])
    return sources


def _format_sources_block(sources: "dict[str, set]") -> str:
    """Baut aus (Dateiname -> Seiten) einen Markdown-Quellenblock (leer, wenn keine)."""
    if not sources:
        return ""
    lines = ["**Quellen:**"]
    for name, pages in sources.items():
        if pages:
            page_str = ", ".join(str(p) for p in sorted(pages))
            lines.append(f"- `{name}`, S. {page_str}")
        else:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def ask_agent(messages, method: "str | ChunkingMethod | None" = None) -> str:
    """Stellt dem Agenten eine Frage und gibt die Antwort als Text zurück.
    Der Kontext (Chat-Verlauf) wird mit der Frage geschickt.

    Args:
        messages: Die Nutzerfrage mit Kontext.
        method: Optionale Chunking-Methode; die Retrieval-Tools suchen dann nur in
            den Chunks dieser Methode. None -> methodenübergreifend.
    """
    if method is not None:
        method = ChunkingMethod.from_value(method)
    token = _request_method.set(method)
    try:
        result = agent.invoke({"messages": messages})
    finally:
        _request_method.reset(token)

    result_messages = result["messages"]
    answer = result_messages[-1].content

    # Fallback: Hat der Agent die Quelle NICHT selbst genannt, hängen wir sie
    # deterministisch aus den tatsächlich abgerufenen Chunks an (Dateiname + Seiten).
    if not _has_source_citation(answer):
        block = _format_sources_block(_collect_retrieved_sources(result_messages))
        if block:
            answer = f"{answer}\n\n{block}"

    return answer
