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
                docs.append({
                    "title": file,
                    "content": f.read()
                })
    return docs

def create_embeddings():
    docs = load_files()

    embedded_data = []
    for doc in docs:
        vector = model.encode(doc["content"]).tolist()

        embedded_data.append({
            "title": doc["title"],
            "content": doc["content"],
            "embedding": vector
        })

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, indent=2)

    print("Local Embeddings Generated Successfully!")

if __name__ == "__main__":
    create_embeddings()
