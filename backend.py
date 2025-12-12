import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

load_dotenv()
my_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=my_key)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.bin")

with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"✓ Loaded {len(metadata)} chunks from FAISS")

def retrieve_context(query, top_k=3):
    """Retrieve most relevant chunks from FAISS"""
    # Generate query embedding
    query_embedding = embedding_model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Get results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):  # Validate index
            meta = metadata[idx]
            # Convert L2 distance to similarity score (0-1)
            score = 1 / (1 + distances[0][i])
            
            results.append({
                "title": meta["title"],
                "chunk_index": meta["chunk_index"],
                "score": score,
                "content": meta["content"]
            })
    
    return results

def format_context(results):
    """Format retrieved chunks for LLM prompt"""
    if not results:
        return "No relevant context found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"Source {i} ({result['title']}, relevance: {result['score']:.2f}):\n{result['content']}"
        )
    return "\n\n".join(context_parts)

def create_prompt(query, context):
    """Create RAG prompt with context and query"""
    prompt = f"""You are a knowledgeable assistant specializing in Islamic topics. Answer the question based on the provided context.

If the context contains relevant information, provide a clear and accurate answer.
If the context doesn't contain enough information, say "I don't have enough information to answer this fully" and provide what you can.

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def ask_question(query, top_k=5, verbose=True):
    """
    Complete RAG pipeline: Retrieve from FAISS + Generate with Gemini
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
    
    # Step 1: Retrieve relevant chunks from FAISS
    results = retrieve_context(query, top_k=top_k)
    
    if verbose:
        print(f"\n🔍 Retrieved {len(results)} relevant chunks:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['title']} (score: {r['score']:.3f})")
    
    # Step 2: Format context
    context = format_context(results)
    
    # Step 3: Create prompt
    prompt = create_prompt(query, context)
    
    if verbose:
        print(f"\n{'='*80}")
        print("Generating answer with Gemini...")
        print(f"{'='*80}")
    
    # Step 4: Generate answer with Gemini
    try:
        response = model.generate_content(prompt)
        answer = response.text
        
        if verbose:
            print(f"\n💡 Answer:\n{answer}\n")
        
        return {
            "query": query,
            "answer": answer,
            "sources": results,
            "context": context
        }
    
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(f"\n❌ {error_msg}\n")
        return {
            "query": query,
            "answer": error_msg,
            "sources": results,
            "context": context
        }

def interactive_chat():
    
    while True:
        query = input("\n📝 Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        ask_question(query, top_k=3, verbose=True)

if __name__ == "__main__":
    # Test single query
    # ask_question("What is salat in Islam?")
    
    # Interactive mode
    interactive_chat()