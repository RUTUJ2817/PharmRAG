import json
import os
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Load models and index lazily
_model = None
_index = None
_chunks = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_index_and_chunks():
    global _index, _chunks
    if _index is None or _chunks is None:
        index_dir = os.path.join("data", "index")
        _index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            _chunks = json.load(f)
    return _index, _chunks

def call_llm(prompt, model_name):
    if "llama" in model_name.lower():
        # Groq
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    else:
        # OpenRouter
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        model_id = "deepseek/deepseek-r1" if "deepseek" in model_name.lower() else "qwen/qwen-3-32b"
        # Openrouter can map Qwen 3 if it's there, else it falls back or errors, but user specified it
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

def grounded_answer(query, model_name):
    model = get_embedding_model()
    index, chunks = get_index_and_chunks()
    
    # 2. Embed query
    query_emb = model.encode([query], convert_to_numpy=True)
    
    # 3. Retrieve top 5 chunks
    k = 5
    distances, indices = index.search(query_emb, k)
    
    retrieved = []
    sources_text = ""
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        retrieved.append(chunk)
        sources_text += f"\n[{chunk['drug']} - {chunk['section']}]:\n{chunk['text']}\n"
    
    # 4. Construct prompt
    prompt = f"""You are a clinical pharmacist.
Answer ONLY using the FDA drug label excerpts below.
If insufficient data, say: 'Insufficient FDA label data.'
Cite sources as [DRUG - section].

FDA Label Excerpts:
{sources_text}

Query: {query}"""

    # 5. Call model
    answer = call_llm(prompt, model_name)
    
    return answer, retrieved

def ungrounded_answer(query, model_name):
    prompt = f"You are a clinical pharmacist. Answer the following query: {query}"
    answer = call_llm(prompt, model_name)
    return answer

if __name__ == "__main__":
    query = "Is warfarin safe with clarithromycin?"
    print(f"Query: {query}")
    print("\n--- Ungrounded (Llama) ---")
    print(ungrounded_answer(query, "Llama 3.3 70B"))
    print("\n--- Grounded (Llama) ---")
    ans, sources = grounded_answer(query, "Llama 3.3 70B")
    print(ans)
    print("\nSources retrieved:", len(sources))
