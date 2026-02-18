from llama_cpp import Llama

llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,  # Use GPU if available (optional)
    verbose=False      # Reduce debug output
)

# Use create_chat_completion for chat format
output = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in simple terms."
        }
    ],
    max_tokens=256,
    temperature=0.2
)

print(output['choices'][0]['message']['content'])