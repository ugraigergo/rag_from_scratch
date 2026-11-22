from rag import save_index, load_index

def test_save_and_load_roundtrip(tmp_path):
    index = [("hello", [0.1, 0.2]), ("world", [0.3, 0.4])]
    path = tmp_path / "index.json"
    save_index(index, str(path))
    loaded = load_index(str(path))
    assert len(loaded) == 2
    assert loaded[0][0] == "hello"
    assert loaded[1][1] == [0.3, 0.4]