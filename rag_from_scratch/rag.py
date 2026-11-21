import requests

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100)->list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size-overlap
    for start in range(0, len(text), step):
        chunk=text[start : start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        
    return chunks


def embed(text: str)->list[float]:
    url = "http://localhost:11434/api/embed"
    payload = {"model": "nomic-embed-text", "input": text}

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    return data["embeddings"][0]
