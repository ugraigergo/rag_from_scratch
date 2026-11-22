from rag import load_text_file, chunk_text, build_index, answer, retrieve, chunk_by_sentences, save_index, load_index
import os

INDEX_PATH = "index.json"

if(os.path.exists(INDEX_PATH)):
    index = load_index(INDEX_PATH)
else:
    text = load_text_file("./resources/doc.txt")
    chunks =chunk_by_sentences(text, max_chars=400)
    index = build_index(chunks)
    save = save_index(index, "./resources/index.json")

print(answer("What does FocusEdge do?", index, top_k = 7))

