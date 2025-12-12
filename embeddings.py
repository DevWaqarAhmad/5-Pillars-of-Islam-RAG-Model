"""
Create embeddings and store in FAISS (Easier to install than ChromaDB)
This replaces embeddings.py
"""

import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_files(folder="data"):
    """Load all text files from data folder"""
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    docs.append({
                        "title": file,
                        "content": content
                    })
    return docs

def chunk_text(text, max_words=150, overlap=30):
    """Create overlapping chunks"""
    words = text.split()
    chunks = []
    
    if len(words) <= max_words:
        return [text]
    
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
        
        if i + max_words >= len(words):
            break
    
    return chunks

def create_embeddings(folder="data"):
    """
    Create embeddings and store in FAISS index
    """
    print("Creating Vector Database with FAISS...")
    print("="*60)
    
    # Load documents
    docs = load_files(folder)
    print(f"✓ Loaded {len(docs)} documents from {folder}/")
    
    # Process documents into chunks
    all_chunks = []
    all_metadata = []
    
    for doc in docs:
        chunks = chunk_text(doc["content"], max_words=150, overlap=30)
        
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "title": doc["title"],
                "chunk_index": idx,
                "content": chunk,
                "total_chunks": len(chunks)
            })
    
    print(f"✓ Created {len(all_chunks)} chunks")
    
    # Generate embeddings
    print("✓ Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    
    # Add embeddings to index
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, "faiss_index.bin")
    
    # Save metadata
    with open("faiss_metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f)
    
    print("\n" + "="*60)
    print("✅ Vector Database Created Successfully!")
    print(f"  📊 Total chunks: {len(all_chunks)}")
    print(f"  📄 Documents: {len(docs)}")
    print(f"  💾 Index file: faiss_index.bin")
    print(f"  💾 Metadata: faiss_metadata.pkl")
    print("="*60)
    
    return index, all_metadata

def add_new_document(file_path):
    """Add a single new document to existing FAISS index"""
    # Load existing index and metadata
    index = faiss.read_index("faiss_index.bin")
    
    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Read new file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content:
        print("Empty file, skipping...")
        return
    
    filename = os.path.basename(file_path)
    
    # Chunk the content
    chunks = chunk_text(content, max_words=150, overlap=30)
    
    # Generate embeddings
    new_embeddings = model.encode(chunks, show_progress_bar=False)
    new_embeddings = np.array(new_embeddings).astype('float32')
    
    # Add to index
    index.add(new_embeddings)
    
    # Add metadata
    for idx, chunk in enumerate(chunks):
        metadata.append({
            "title": filename,
            "chunk_index": idx,
            "content": chunk,
            "total_chunks": len(chunks)
        })
    
    # Save updated index and metadata
    faiss.write_index(index, "faiss_index.bin")
    
    with open("faiss_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Added {len(chunks)} chunks from {filename}")
    print(f"   Total chunks in DB: {len(metadata)}")

def get_stats():
    """Get statistics about the FAISS index"""
    try:
        index = faiss.read_index("faiss_index.bin")
        
        with open("faiss_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        print(f"\n📊 Vector Database Stats:")
        print(f"  Total chunks: {len(metadata)}")
        print(f"  Index size: {index.ntotal}")
        print(f"  ✓ Database is working correctly")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("   Database doesn't exist. Run create_embeddings() first.")

if __name__ == "__main__":
    # Create embeddings from data folder
    create_embeddings(folder="data")
    
    # Show stats
    print("\n")
    get_stats()
    
    print("\n💡 Next Steps:")
    print("   1. Use backend_faiss.py to query the database")
    print("   2. Much faster than JSON!")
    print("   3. Add new documents anytime with add_new_document()")