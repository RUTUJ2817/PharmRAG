import requests
import json
import os
import time

DRUGS = [
    "ibuprofen", "warfarin", "metformin", "metoprolol", "amoxicillin",
    "sertraline", "atorvastatin", "lisinopril", "omeprazole",
    "azithromycin", "levothyroxine", "amlodipine", "albuterol",
    "prednisone", "clarithromycin"
]

FIELDS = [
    "indications_and_usage",
    "contraindications",
    "warnings",
    "dosage_and_administration",
    "drug_interactions",
    "pregnancy",
    "adverse_reactions"
]

def download_labels():
    for drug in DRUGS:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug}&limit=1"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    extracted = {}
                    for field in FIELDS:
                        # Some fields might not exist or might be under slightly different names
                        # The FDA API returns lists of strings for each section
                        content = ""
                        if field in result:
                            content = " ".join(result[field])
                        elif field == "pregnancy" and "pregnancy_or_breast_feeding" in result:
                            content = " ".join(result["pregnancy_or_breast_feeding"])
                        
                        # Truncate to ~2000 chars
                        if len(content) > 2000:
                            content = content[:1997] + "..."
                        
                        extracted[field] = content
                    
                    filepath = os.path.join("data", "drug_labels", f"{drug}.json")
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(extracted, f, indent=2)
                    print(f"Saved {drug}")
            else:
                print(f"Failed to fetch {drug}: {response.status_code}")
        except Exception as e:
            print(f"Error fetching {drug}: {e}")
        
        # Be nice to FDA API
        time.sleep(0.5)

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "drug_labels"), exist_ok=True)
    download_labels()
