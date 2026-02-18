from llama_cpp import Llama

llm = Llama(
    model_path="models/mistral-7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,   # set to physical cores
)

output = llm(
    "Explain RAG in simple terms.",
    max_tokens=256,
    temperature=0.2,
)

print(output["choices"][0]["text"])
