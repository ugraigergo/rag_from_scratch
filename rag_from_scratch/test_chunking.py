from rag import chunk_text, embed, cosine_similarity

def test_short_text_single_chunk():
    assert chunk_text("hello", 500, 100) == ["hello"]

def test_overlap():
    chunks = chunk_text("a" * 1000, 500, 100)
    assert chunks[0][-100:] == chunks[1][:100]

def test_no_text_lost():
    text = "x" * 1234
    chunks = chunk_text(text, 500, 100)
    assert len(chunks[0]) == 500
    assert chunks[-1].endswith(text[-10:])

def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)

def test_no_redundant_last_chunk():
    chunks = chunk_text("a" * 500, 500, 100)
    assert len(chunks) == 1

def test_embed_length():
    assert len(embed("hello"))==768