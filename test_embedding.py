import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load embeddings
with open("embeddings.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search(query, top_k=3):
    query_vec = model.encode(query).tolist()
    
    scores = []
    for doc in embedded_data:
        score = cosine_similarity(query_vec, doc["embedding"])
        scores.append((doc["title"], score, doc["content"]))
    
    # Remove duplicates (keep highest score per document)
    results_dict = {}
    for title, score, content in scores:
        if title not in results_dict or score > results_dict[title][0]:
            results_dict[title] = (score, content)
    
    unique_results = [(title, score, content) for title, (score, content) in results_dict.items()]
    unique_results.sort(key=lambda x: x[1], reverse=True)
    
    return unique_results[:top_k]

# --- Multiple keyword query ---
queries = ["salat", "swam"]
combined_results = {}

for q in queries:
    results = search(f"What is {q} in Islam?", top_k=3)
    for title, score, content in results:
        if title not in combined_results or score > combined_results[title][0]:
            combined_results[title] = (score, content)

# Sort final combined results
final_results = sorted([(t, s, c) for t, (s, c) in combined_results.items()], key=lambda x: x[1], reverse=True)

# Print results
for title, score, content in final_results:
    print(f"Title: {title}, Score: {score:.4f}")
    print(f"Content: {content[:300]}...\n")
