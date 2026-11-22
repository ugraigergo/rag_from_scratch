from rag import chunk_by_sentences

def test_chunks_respect_max_chars():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_by_sentences(text, max_chars=30)
    assert all(len(c) <= 30 for c in chunks)

def test_chunks_end_at_sentence_boundary():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_by_sentences(text, max_chars=30)
    assert all(c.endswith(".") for c in chunks)

def test_long_sentence_kept_whole():
    long_sentence = "word " * 20 + "end."
    assert chunk_by_sentences(long_sentence, max_chars=30) == [long_sentence]

def test_empty():
    assert chunk_by_sentences("") == []