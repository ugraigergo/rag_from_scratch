from rag import split_sentences

def test_basic_split():
    assert split_sentences("Hello world. How are you? Fine!") == ["Hello world.", "How are you?", "Fine!"]

def test_empty():
    assert split_sentences("") == []

def test_no_punctuation():
    assert split_sentences("no end here") == ["no end here"]