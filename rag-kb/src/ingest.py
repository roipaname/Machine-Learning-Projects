import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma as Chroma

from src.loader import load_documents

VECTORSTORE_PATH = "data/vectorstore"


def ingest():
    docs = load_documents("./data/documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=VECTORSTORE_PATH
        )

    vectorstore.persist()


if __name__ == "__main__":
    ingest()
