from rag import build_prompt

def test_prompt_contains_query_and_context():
    prompt = build_prompt("What is X?", ["chunk one", "chunk two"])
    assert "What is X?" in prompt
    assert "chunk one" in prompt
    assert "chunk two" in prompt