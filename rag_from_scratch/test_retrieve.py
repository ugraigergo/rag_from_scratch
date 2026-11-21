from rag import build_index, retrieve

def test_retrieve_finds_relevant_chunk():
    chunks = [
        "The recipe needs two eggs and a cup of flour.",
        "Backpropagation updates the weights of a deep learning model.",
        "The football match ended with a 2-1 score.",
    ]
    index = build_index(chunks)
    results = retrieve("How do neural networks learn?", index, top_k=1)
    assert len(results) == 1
    assert "Backpropagation" in results[0][1]