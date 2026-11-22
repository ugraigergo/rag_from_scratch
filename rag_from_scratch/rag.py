import requests
import numpy as np

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

def cosine_similarity(a: list[float], b: list[float])->float:
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot/(norm_a*norm_b)

def build_index(chunks: list[str])->list[tuple[str, list[float]]]:
    embedded_chunks = []
    for chunk in chunks:
        vektor = embed(chunk)
        x = (chunk, vektor)
        embedded_chunks.append(x)

    return embedded_chunks

def retrieve(query: str, index: list[tuple[str, list[float]]], top_k: int = 3) -> list[tuple[float, str]]:
    q_vec=embed(query)
    scored = []
    for chunk, vector in index:
        score = cosine_similarity(q_vec, vector)
        x = (score, chunk)
        scored.append(x)
    scored.sort(reverse=True)

    return scored[:top_k]

def build_prompt(query: str, context_chunks:list[str])->str:
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using ONLY the context below.
If the answer is not in the context, say I don't know."

Context:
{context}

Question: {query}
Answer:"""

def generate(prompt: str)->str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]

def answer(query: str, index, top_k: int = 3)-> str:
    results = retrieve(query, index, top_k)
    context_chunks = [chunk for score, chunk in results]
    prompt = build_prompt(query, context_chunks)

    return generate(prompt)


def load_text_file(path: str)->str:
    with open(path, encoding="utf-8") as f:

        return f.read()

import re
def split_sentences(text: str)->list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_by_sentences(text: str, max_chars: int = 400)->list[str]:
    sentences=split_sentences(text)
    chunks = []
    current = ""
    for sentence in sentences:
        if current and len(sentence) + len(current) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)

    return chunks