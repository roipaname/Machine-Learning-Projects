from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from prompts import QA_PROMPT

VECTORSTORE_PATH = "data/vectorstore"

def ask(question:str):
    embeddings=OpenAIEmbeddings()

    vectorstore=FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    qa=RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini",temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k":4}),
        chain_type_kwargs={"prompt":QA_PROMPT}
    )
    return qa.run(question)

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print(ask(q))