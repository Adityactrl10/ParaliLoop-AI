
import json
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "boundaries"
OUTPUT_FILE = OUTPUT_DIR / "india_districts.geojson"

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Getting India district boundary information...")

    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "ParaliLoop-AI"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))

    if isinstance(metadata, list):
        metadata = metadata[0]

    download_url = metadata.get("simplifiedGeometryGeoJSON")

    if not download_url:
        raise RuntimeError("Simplified GeoJSON download link not found.")

    print("Downloading district boundaries...")

    urllib.request.urlretrieve(download_url, OUTPUT_FILE)

    print("Download completed!")
    print("Saved at:", OUTPUT_FILE)
    print("Boundary year:", metadata.get("boundaryYearRepresented"))
    print("Source:", metadata.get("boundarySource"))
    print("License:", metadata.get("boundaryLicense"))

if __name__ == "__main__":
    main()
