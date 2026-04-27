import json
import os
import time
from query import grounded_answer, ungrounded_answer

QUERIES = [
    {"id": "q1", "category": "drug interactions", "text": "Is warfarin safe with clarithromycin?"},
    {"id": "q2", "category": "drug interactions", "text": "Can I take ibuprofen if I am on omeprazole and atorvastatin?"},
    {"id": "q3", "category": "pregnancy safety", "text": "I am 32 weeks pregnant. Can I take ibuprofen for a headache?"},
    {"id": "q4", "category": "pregnancy safety", "text": "Is amoxicillin safe during the first trimester of pregnancy?"},
    {"id": "q5", "category": "CKD dosing", "text": "What is the recommended dose of metformin for a patient with eGFR of 25?"},
    {"id": "q6", "category": "CKD dosing", "text": "Does lisinopril need renal dose adjustment?"},
    {"id": "q7", "category": "contraindications", "text": "Is metoprolol contraindicated in patients with asthma?"},
    {"id": "q8", "category": "contraindications", "text": "Can a patient with active liver disease take atorvastatin?"},
    {"id": "q9", "category": "drug interactions", "text": "What happens if I mix sertraline and ibuprofen?"},
    {"id": "q10", "category": "adverse reactions", "text": "What are the severe adverse reactions of azithromycin?"}
]

MODELS = ["Llama 3.3 70B", "DeepSeek R1", "Qwen 3 32B"]

def run_comparison():
    results = []
    
    os.makedirs(os.path.join("data", "results"), exist_ok=True)
    
    for q in QUERIES:
        for model in MODELS:
            print(f"Processing query: '{q['text']}' with model: {model}")
            
            try:
                ungrounded = ungrounded_answer(q["text"], model)
                time.sleep(1) # Rate limit
                grounded, _ = grounded_answer(q["text"], model)
                time.sleep(1) # Rate limit
            except Exception as e:
                print(f"Error during query execution: {e}")
                ungrounded = "Error"
                grounded = "Error"
            
            results.append({
                "query_id": q["id"],
                "category": q["category"],
                "query": q["text"],
                "model": model,
                "ungrounded": ungrounded,
                "grounded": grounded
            })
            
    filepath = os.path.join("data", "results", "comparison.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {filepath}")

if __name__ == "__main__":
    run_comparison()
