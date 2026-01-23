from langchain.prompts import PromptTemplate

QA_PROMPT=PromptTemplate(
    input_variables=["context","question"],
    template="""
You are a knowledge base assistant.
Answer ONLY using the provided context.
If the answer is not present, say: "I don't know."

Context:
{context}

Question:
{question}
"""
)