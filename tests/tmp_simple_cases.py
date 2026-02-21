import json
import requests

QUERY = "Write a java code that finds the greatest number in [25,12,3,56] using binary search"

payload = {
    "query": QUERY,
    "language": "python",
    "context": {},
}

print(f"\nQuery: {QUERY}")
print("=" * 60)

resp = requests.post("http://127.0.0.1:8080/council/run", json=payload, timeout=600)
print(f"HTTP {resp.status_code}")
data = resp.json()

print(f"status       : {data.get('status')}")
print(f"passes       : {data.get('refinement_passes')}")
print(f"agents_used  : {data.get('agents_used')}")
print(f"duration     : {data.get('total_duration_secs')}s")
print(f"summary      : {data.get('summary', '')[:300]}")

code = data.get("code") or {}
files = code.get("files") or []
print(f"files        : {len(files)}")

for i, f in enumerate(files):
    print(f"\n{'─'*60}")
    print(f"FILE {i+1}: {f.get('filename','?')}  ({f.get('language','?')})")
    print(f"{'─'*60}")
    print(f"{f.get('content','(empty)')}")
    print(f"{'─'*60}")

if not files:
    print("\n⚠ NO CODE FILES RETURNED")
    print("Raw 'code' field:")
    print(json.dumps(code, indent=2)[:2000])
