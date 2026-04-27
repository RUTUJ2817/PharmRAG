import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index():
    labels_dir = os.path.join("data", "drug_labels")
    index_dir = os.path.join("data", "index")
    os.makedirs(index_dir, exist_ok=True)
    
    chunks = []
    chunk_size = 400
    
    # 1. Load labels and chunk text
    for filename in os.listdir(labels_dir):
        if not filename.endswith(".json"):
            continue
            
        drug_name = filename.replace(".json", "")
        filepath = os.path.join(labels_dir, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for section, content in data.items():
            if not content:
                continue
            
            # Simple chunking by character length
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) > 20: # skip very small/empty chunks
                    chunks.append({
                        "drug": drug_name,
                        "section": section,
                        "text": chunk_text
                    })
                    
    if not chunks:
        print("No chunks created. Are the labels downloaded?")
        return

    print(f"Created {len(chunks)} chunks.")
    
    # 2. Embed
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [c["text"] for c in chunks]
    print("Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # 3. Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 4. Save
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
        
    print("Index and chunks saved successfully.")

if __name__ == "__main__":
    build_index()
