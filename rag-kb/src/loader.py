from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
import logging
logging.basicConfig(logging.INFO)
def load_documents(path:str):
    docs=[]
    if not path:
        logging.error("Path is empty")
        raise


    for file in Path(path).iterdir():
        if file.suffix ==".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix==".txt":
            docs.extend(TextLoader(str(file)).load())
        elif file.suffix==".docx":
            docs.extend(Docx2txtLoader(str(file)).load())

    return docs
