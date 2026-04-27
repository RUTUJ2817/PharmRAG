import json
import os
import time
from query_elite import grounded_answer_elite, ungrounded_answer_elite

MODELS = ["Llama 3.3 70B", "DeepSeek R1", "Qwen 3 32B"]

def run_comparison_elite():
    with open(os.path.join("data", "queries_elite.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    results = []
    os.makedirs(os.path.join("data", "results"), exist_ok=True)
    
    for q in queries:
        if q['id'] not in ["q_L1_1", "q_L1_2", "q_L1_3", "q_L2_1", "q_L2_4", "q_L2_6", "q_L3_1", "q_L3_5", "q_L3_9"]:
            continue
        for model in MODELS:
            print(f"Processing elite query {q['id']} with model: {model}")
            
            try:
                ungrounded = ungrounded_answer_elite(q["text"], model)
                time.sleep(1)
                grounded, retrieved_chunks = grounded_answer_elite(q["text"], model)
                time.sleep(1)
            except Exception as e:
                print(f"Error during query execution: {e}")
                ungrounded = "Error"
                grounded = "Error"
                retrieved_chunks = []
                
            results.append({
                "query_id": q["id"],
                "category": q["category"],
                "level": q["level"],
                "query": q["text"],
                "model": model,
                "ungrounded": ungrounded,
                "grounded": grounded,
                "retrieved_chunks": retrieved_chunks
            })
            
    filepath = os.path.join("data", "results", "comparison_elite.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved elite results to {filepath}")

if __name__ == "__main__":
    run_comparison_elite()
