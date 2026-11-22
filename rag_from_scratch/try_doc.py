from rag import load_text_file, chunk_text, build_index, answer, retrieve, chunk_by_sentences

text = load_text_file("./resources/doc.txt")
chunks = chunk_by_sentences(text, max_chars=400)
print(len(chunks), "chunks")
for c in chunks:
    print(repr(c[:60]))
index = build_index(chunks)

print(answer("What projects has Gergő built?", index, top_k=7))