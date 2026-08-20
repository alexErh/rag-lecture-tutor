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
    chunk_size=100,
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


def add_document(path: str):
    chunks = chunk_file(path)

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )


def retrieve_chunks(query: str, n: int = 3):
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