import requests, os
from dotenv import load_dotenv
load_dotenv()
r = requests.get('https://openrouter.ai/api/v1/models', headers={'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY","")}'}, timeout=10)
models = r.json().get('data',[])
free = [m for m in models if 'deepseek' in m.get('id','').lower() and m.get('pricing',{}).get('prompt','1') == '0']
for m in free:
    print(m['id'])
print("---FREE QWEN---")
free2 = [m for m in models if 'qwen' in m.get('id','').lower() and m.get('pricing',{}).get('prompt','1') == '0']
for m in free2:
    print(m['id'])
# Also check credit balance
print("---CREDITS---")
cr = requests.get('https://openrouter.ai/api/v1/credits', headers={'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY","")}'}, timeout=10)
print(cr.json())
