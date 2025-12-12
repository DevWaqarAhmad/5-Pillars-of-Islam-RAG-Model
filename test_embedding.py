import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load embeddings
with open("embeddings.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)

print(f"Loaded {len(embedded_data)} embedded chunks")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product > 0 else 0.0

def search_chunks(query, top_k=5, min_score=0.3):
    """
    Search for most relevant chunks (not documents)
    Returns top_k chunks with scores above min_score
    """
    query_vec = model.encode(query, show_progress_bar=False).tolist()
    
    results = []
    for doc in embedded_data:
        score = cosine_similarity(query_vec, doc["embedding"])
        if score >= min_score:  # Filter low-quality matches
            results.append({
                "title": doc["title"],
                "chunk_index": doc["chunk_index"],
                "score": score,
                "content": doc["content"]
            })
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def format_context(results):
    """
    Format retrieved chunks into context for LLM
    """
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {result['title']} (chunk {result['chunk_index']}, relevance: {result['score']:.3f})]\n"
            f"{result['content']}\n"
        )
    return "\n".join(context_parts)

# --- Test Queries ---
if __name__ == "__main__":
    test_queries = [
        "What is salat in Islam?",
        "Tell me about fasting",
        "What are the pillars of Islam?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = search_chunks(query, top_k=3, min_score=0.2)
        
        if not results:
            print("No relevant results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n[Result {i}] {result['title']} (chunk {result['chunk_index']})")
                print(f"Score: {result['score']:.4f}")
                print(f"Content: {result['content'][:200]}...")
                print("-" * 80)
            
            # Print formatted context for RAG
            print("\n--- FORMATTED CONTEXT FOR LLM ---")
            print(format_context(results))