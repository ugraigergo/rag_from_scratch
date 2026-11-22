from rag import file_fingerprint

def test_same_content_same_fingerprint(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("hello")
    assert file_fingerprint(str(a)) == file_fingerprint(str(b))

def test_changed_content_changes_fingerprint(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello")
    before = file_fingerprint(str(f))
    f.write_text("hello!")
    after = file_fingerprint(str(f))
    assert before != after