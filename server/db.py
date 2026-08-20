# ========= IMPORTS =========
import chromadb
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# ========= GLOBAL =========
client = chromadb.PersistentClient(path='./VectorDB')

collection = client.get_or_create_collection(
    name="VectorDB",
)

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
    chunks = simple_chunker.split_documents([document])
    return chunks

def add_document(path: str):
    chunks = chunk_file(path)
    for chunk in chunks:
        #print("=== "+chunk.page_content+'\n')
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=chunk.page_content
        )


def retrieve_chunks(query: str, n: int = 3):
    return collection.query(query_texts=[query], n_results=n)['documents']

def reset_db():
    client.delete_collection(collection.name)

if __name__ == '__main__':

    add_document('data/processed/example.md')
    print(retrieve_chunks('Wie könnnen sich Mitarbeitende Weiterbilden?'))