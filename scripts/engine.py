"""
PharmRAG — Core LLM + Retrieval Engine
Handles all model calls, embedding, indexing, and retrieval logic.
"""

import json
import os
import time
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Registry ──────────────────────────────────────────────────
MODELS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 4 Scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Qwen 3 32B": "qwen/qwen3-32b",
}

# ── Lazy-loaded singletons ──────────────────────────────────────────
_dense_model = None
_cross_encoder = None
_nli_model = None
_index = None
_chunks = None
_bm25 = None


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _dense_model


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _get_index_and_chunks():
    global _index, _chunks, _bm25
    if _index is None:
        idx_dir = os.path.join("data", "index")
        _index = faiss.read_index(os.path.join(idx_dir, "index.faiss"))
        with open(os.path.join(idx_dir, "chunks.json"), "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        tokenized = [c["text"].lower().split() for c in _chunks]
        _bm25 = BM25Okapi(tokenized)
    return _index, _chunks, _bm25


# ── LLM Call with retry ─────────────────────────────────────────────
def call_llm(prompt: str, model_name: str, max_retries: int = 4) -> str:
    groq_id = MODELS.get(model_name)
    if not groq_id:
        raise ValueError(f"Unknown model: {model_name}")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": groq_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited ({model_name}), waiting {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip <think> tags from reasoning models
            if "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited ({model_name}), waiting {wait}s…")
                time.sleep(wait)
                continue
            print(f"  HTTP error calling {model_name}: {e}")
            return f"[API Error: {resp.status_code}]"
        except Exception as e:
            print(f"  Error calling {model_name}: {e}")
            time.sleep(2)
    return "[API Error: max retries exceeded]"


# ── Retrieval Strategies ────────────────────────────────────────────
def retrieve_dense(query: str, k: int = 5) -> list:
    """Naive dense-only retrieval (baseline)."""
    model = _get_dense_model()
    index, chunks, _ = _get_index_and_chunks()
    q_emb = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]


def retrieve_hybrid(query: str, k: int = 5) -> list:
    """BM25 + Dense + Cross-Encoder reranking."""
    dense_model = _get_dense_model()
    cross_encoder = _get_cross_encoder()
    index, chunks, bm25 = _get_index_and_chunks()

    # Dense top-20
    q_emb = dense_model.encode([query], convert_to_numpy=True)
    _, d_idx = index.search(q_emb, 20)
    dense_hits = {i: chunks[i] for i in d_idx[0]}

    # BM25 top-20
    sparse_scores = bm25.get_scores(query.lower().split())
    s_idx = np.argsort(sparse_scores)[::-1][:20]
    sparse_hits = {int(i): chunks[int(i)] for i in s_idx}

    # Merge unique candidates
    merged = {**dense_hits, **sparse_hits}
    candidates = list(merged.values())

    # Cross-encoder rerank
    pairs = [[query, c["text"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = np.argsort(scores)[::-1]
    return [candidates[i] for i in ranked[:k]]


# ── Answer Generation ──────────────────────────────────────────────
GROUNDED_SYSTEM = """You are a clinical pharmacist.
Answer ONLY using the FDA drug label excerpts below.
If insufficient data, say: 'Insufficient FDA label data.'
Cite sources as [DRUG — SECTION]."""


def answer_ungrounded(query: str, model: str) -> str:
    return call_llm(
        f"You are a clinical pharmacist. Answer the following query:\n\n{query}",
        model,
    )


def answer_grounded(query: str, model: str, retriever: str = "hybrid") -> tuple:
    if retriever == "dense":
        chunks = retrieve_dense(query)
    else:
        chunks = retrieve_hybrid(query)

    ctx = "\n".join(
        f"[{c['drug']} — {c['section']}]:\n{c['text']}" for c in chunks
    )
    prompt = f"{GROUNDED_SYSTEM}\n\nFDA Label Excerpts:\n{ctx}\n\nQuery: {query}"
    return call_llm(prompt, model), chunks


# ── NLI Consistency Check ──────────────────────────────────────────
def check_nli(answer: str, chunks: list) -> dict:
    """Check factual consistency sentence-by-sentence against retrieved context."""
    nli = _get_nli_model()
    context = " ".join(c["text"] for c in chunks)

    # Split answer into sentences for granular analysis
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 15]
    if not sentences:
        return {"label": "Neutral", "entailed": 0, "contradicted": 0, "neutral": 0}

    entailed = contradicted = neutral = 0
    for sent in sentences:
        scores = nli.predict([(context, sent)])
        pred = int(np.argmax(scores[0]))
        if pred == 0:
            contradicted += 1
        elif pred == 1:
            entailed += 1
        else:
            neutral += 1

    total = len(sentences)
    if contradicted / total > 0.3:
        label = "Contradiction"
    elif entailed / total > 0.4:
        label = "Entailment"
    else:
        label = "Neutral"

    return {"label": label, "entailed": entailed, "contradicted": contradicted, "neutral": neutral}
