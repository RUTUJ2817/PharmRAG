import json
import os
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Load models and index lazily
_dense_model = None
_cross_encoder = None
_index = None
_chunks = None
_bm25 = None

def get_models():
    global _dense_model, _cross_encoder
    if _dense_model is None:
        _dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        # We use a fast, small cross encoder
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _dense_model, _cross_encoder

def get_indices():
    global _index, _chunks, _bm25
    if _index is None or _chunks is None:
        index_dir = os.path.join("data", "index")
        _index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            _chunks = json.load(f)
            
        # Build BM25 index
        tokenized_corpus = [chunk["text"].lower().split() for chunk in _chunks]
        _bm25 = BM25Okapi(tokenized_corpus)
        
    return _index, _chunks, _bm25

def call_llm(prompt, model_name):
    if "llama" in model_name.lower():
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        model_id = "deepseek/deepseek-r1" if "deepseek" in model_name.lower() else "qwen/qwen-2.5-32b-instruct"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return "Error: Could not retrieve answer."

def retrieve_hybrid(query, k=5):
    dense_model, cross_encoder = get_models()
    index, chunks, bm25 = get_indices()
    
    # 1. Dense retrieval (top 15)
    query_emb = dense_model.encode([query], convert_to_numpy=True)
    distances, dense_indices = index.search(query_emb, 15)
    dense_results = [chunks[idx] for idx in dense_indices[0]]
    
    # 2. Sparse retrieval (top 15)
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(sparse_scores)[::-1][:15]
    sparse_results = [chunks[idx] for idx in sparse_indices]
    
    # 3. Combine unique candidates
    candidates_dict = {}
    for chunk in dense_results + sparse_results:
        # Use drug + section + first 50 chars as unique key
        key = f"{chunk['drug']}_{chunk['section']}_{chunk['text'][:50]}"
        candidates_dict[key] = chunk
        
    candidates = list(candidates_dict.values())
    
    # 4. Cross-Encoder Re-ranking
    cross_inp = [[query, chunk["text"]] for chunk in candidates]
    cross_scores = cross_encoder.predict(cross_inp)
    
    # Sort by cross-encoder score
    ranked_indices = np.argsort(cross_scores)[::-1]
    
    # Return top K
    top_k_chunks = [candidates[idx] for idx in ranked_indices[:k]]
    return top_k_chunks

def grounded_answer_elite(query, model_name):
    retrieved = retrieve_hybrid(query, k=5)
    
    sources_text = ""
    for chunk in retrieved:
        sources_text += f"\n[{chunk['drug']} - {chunk['section']}]:\n{chunk['text']}\n"
    
    prompt = f"""You are a clinical pharmacist.
Answer ONLY using the FDA drug label excerpts below.
If insufficient data, say: 'Insufficient FDA label data.'
Cite sources as [DRUG - section].

FDA Label Excerpts:
{sources_text}

Query: {query}"""

    answer = call_llm(prompt, model_name)
    return answer, retrieved

def ungrounded_answer_elite(query, model_name):
    prompt = f"You are a clinical pharmacist. Answer the following query: {query}"
    return call_llm(prompt, model_name)

if __name__ == "__main__":
    q = "Is warfarin safe with clarithromycin?"
    print(grounded_answer_elite(q, "Llama 3.3 70B")[0])
