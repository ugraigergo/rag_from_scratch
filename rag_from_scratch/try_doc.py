from rag import load_text_file, chunk_text, build_index, answer, retrieve, chunk_by_sentences

text = load_text_file("./resources/doc.txt")
chunks = chunk_by_sentences(text, max_chars=400)
print(len(chunks), "chunks")
index = build_index(chunks)

questions = [
    "What does FocusEdge do?",
    "Which certification does Gergő have?",
    "What stack was used for the BSc thesis?",
    "What is Gergő's favorite food?",
]
for q in questions:
    print(q)
    print(answer(q, index, top_k=7))
    print()