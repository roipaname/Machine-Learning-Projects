import os

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss as FAISS

from loader import load_documents

VECTORSTORE_PATH = "data/vectorstore"


def ingest():

    docs=load_documents("./data/documents")
    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    chunks=splitter.split_documents(docs)
    embeddings= OpenAIEmbeddings()

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore=FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore=FAISS.from_documents(chunks,embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)

if __name__=="__main__":
    ingest()

