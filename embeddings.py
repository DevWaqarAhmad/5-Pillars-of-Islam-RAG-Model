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
                content = f.read().strip()
                if content:  # Skip empty files
                    docs.append({
                        "title": file,
                        "content": content
                    })
    return docs

def chunk_text(text, max_words=150, overlap=30):
    """
    Create overlapping chunks for better context preservation
    """
    words = text.split()
    chunks = []
    
    if len(words) <= max_words:
        return [text]
    
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
        
        # Break if we've covered the text
        if i + max_words >= len(words):
            break
    
    return chunks

def create_embeddings(folder="data"):
    docs = load_files(folder)
    embedded_data = []

    for doc in docs:
        chunks = chunk_text(doc["content"], max_words=150, overlap=30)
        for idx, chunk in enumerate(chunks):
            vector = model.encode(chunk, show_progress_bar=False).tolist()
            embedded_data.append({
                "title": doc["title"],
                "chunk_index": idx,
                "content": chunk,
                "embedding": vector
            })

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Embeddings Generated: {len(embedded_data)} chunks from {len(docs)} documents")

if __name__ == "__main__":
    create_embeddings()