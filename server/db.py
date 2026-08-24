import chromadb
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ========= CHROMA =========

client = chromadb.PersistentClient(path="./VectorDB")

collection = client.get_or_create_collection(
    name="VectorDB",
)


# ========= CHUNKER =========

simple_chunker = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " "]
)


# ========= FUNCTIONS =========

def chunk_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    document = Document(
        page_content=text,
        metadata={"source": path}
    )

    return simple_chunker.split_documents([document])

# Lese Path von Markdown ein um es Chunken zu lassen und in die DB zu speichern
def add_document(path: str):
    chunks = chunk_file(path)
    for chunk in chunks:
        print('\n ======== Chunk ========')
        print(chunk.page_content)
        print(chunk.metadata)
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


# ========= API =========

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    n: int = 3

class FileRequest(BaseModel):
    filename: str


@app.post("/query")
def query(request: QueryRequest):
    results = retrieve_chunks(
        request.query,
        request.n
    )
    print(results["metadatas"])
    return {
        "documents": results["documents"],
        "distances": results["distances"],
        "metadatas": results["metadatas"],
    }
@app.post("/document")
def get_filechunks(request: FileRequest):
    results = retrieve_through_metadata(request.filename)
    print({
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    })
    return {
        "documents": results["documents"],
        "metadatas": results["metadatas"],
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