from rag import load_text_file, chunk_text, build_index, answer, retrieve

text = load_text_file("./resources/doc.txt")
chunks = chunk_text(text, chunk_size=300, overlap=50)
print(len(chunks), "chunks")
index = build_index(chunks)
results = retrieve("What projects has Gergő built?", index, top_k=6)
for score, chunk in results:
    print(f"{score:.3f}  {chunk[:80]!r}")

print(answer("What projects has Gergő built?", index, top_k=6))