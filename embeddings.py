import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_files(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({
                    "title": file,
                    "content": content
                })
    return docs

def chunk_text(text, max_words=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def create_embeddings(folder="data"):
    docs = load_files(folder)
    embedded_data = []

    for doc in docs:
        chunks = chunk_text(doc["content"], max_words=100)
        for idx, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            embedded_data.append({
                "title": doc["title"],
                "chunk_index": idx,
                "content": chunk,
                "embedding": vector
            })

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, indent=2)

    print("Chunked Embeddings Generated Successfully!")

if __name__ == "__main__":
    create_embeddings()
