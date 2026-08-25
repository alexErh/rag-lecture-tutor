"""Agent-Code: Retrieval-Tools, Modellwahl (lokal/Cloud) und der RAG-Agent.

Der Agent wird beim Import (und damit beim API-Start) aufgebaut. Er nutzt
Retrieval-Tools, die DIREKT in-process auf die DB zugreifen (kein HTTP-Umweg).
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from database import retrieve_chunks, retrieve_through_metadata

# .env laden (LOCAL, LOCAL_BASE_URL, LOCAL_MODEL_NAME, NVIDIA_API_KEY)
load_dotenv()


SYSTEM_PROMPT = """
Du bist ein Tutor für Vorlesungsinhalte.

Beantworte die Frage ausschließlich anhand der Informationen,
die du aus den Retrieval-Tools erhältst.

Wenn die Frage Informationen aus den Vorlesungsunterlagen benötigt,
musst du zuerst das passende Retrieval-Tool verwenden.

WICHTIG:
- Nach einem Tool-Aufruf musst du das Tool-Ergebnis auswerten.
- Gib niemals den Tool-Aufruf selbst als Antwort an den Benutzer.
- Gib niemals die interne Tool-Ausgabe oder deren JSON-Struktur als Antwort aus.
- Formuliere stattdessen eine normale, verständliche Antwort.
- Wenn der Kontext die Frage nicht beantwortet, sage ausdrücklich,
  dass die Information in den bereitgestellten Unterlagen nicht gefunden wurde.
- Erfinde keine Informationen.

Wenn du Informationen aus `search_lecture_docs` verwendest,
nenne anschließend die verwendete Quelle im Format:

*Quelle*: `data/raw/datei.txt`
"""

# LOCAL=true -> lokales Ollama-Modell; LOCAL=false -> Nvidia NIM API.
LOCAL = os.getenv("LOCAL", "true").strip().lower() in ("1", "true", "yes", "ja")
print(LOCAL)

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
            base_url=os.getenv("LOCAL_BASE_URL"),
            model=os.getenv("LOCAL_MODEL_NAME"),
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
