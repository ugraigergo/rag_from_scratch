from rag import build_index, answer

chunks = [
    "Gergő is studying AI at the University of Debrecen.",
    "The Eiffel Tower is in Paris and was completed in 1889.",
    "Ollama lets you run large language models locally.",
]
index = build_index(chunks)

print(answer("Where is Gergő studying?", index, top_k=1))
print(answer("What is the capital of Mars?", index, top_k=1))