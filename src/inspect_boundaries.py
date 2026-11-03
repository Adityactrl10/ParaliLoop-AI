import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILE = BASE_DIR / "data" / "boundaries" / "india_districts.geojson"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Metadata:")
print(json.dumps(data.get("metadata", {}), indent=2))

print("\nFirst feature properties:")
print(json.dumps(data["features"][0]["properties"], indent=2))