from rag import cosine_similarity

def test_identical_vectors():
    assert abs(cosine_similarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9

def test_orthogonal_vectors():
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9

def test_opposite_vectors():
    assert abs(cosine_similarity([1, 0], [-1, 0]) + 1.0) < 1e-9