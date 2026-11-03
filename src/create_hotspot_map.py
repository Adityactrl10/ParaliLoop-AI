
import pandas as pd
import folium
from pathlib import Path
from folium.plugins import HeatMap

# ==========================================
# PARALILOOP AI - HOTSPOT MAP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "cleaned_fires.csv"
)

OUTPUT_DIR = BASE_DIR / "dashboard" / "maps"
OUTPUT_FILE = OUTPUT_DIR / "punjab_haryana_hotspots.html"


def main():

    print("\nLoading NASA fire detections...")

    if not INPUT_FILE.exists():
        print("ERROR: Cleaned dataset not found!")
        return

    data = pd.read_csv(INPUT_FILE)

    data = data.dropna(
        subset=["latitude", "longitude", "acq_date"]
    )

    data["acq_date"] = pd.to_datetime(
        data["acq_date"], errors="coerce"
    )

    data = data.dropna(subset=["acq_date"])

    seasonal = data[
        data["acq_date"].dt.month.isin([10, 11])
    ].copy()

    if seasonal.empty:
        print("No seasonal detections found!")
        return

    print("Seasonal detections:", len(seasonal))

    # OpenStreetMap background
    map_object = folium.Map(
        location=[30.2, 75.5],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    heat_data = seasonal[
        ["latitude", "longitude"]
    ].values.tolist()

    HeatMap(
        heat_data,
        radius=8,
        blur=12,
        min_opacity=0.35
    ).add_to(map_object)

    title_html = """
    <h3 align="center">
    ParaliLoop AI - NASA Fire Detection Hotspots
    </h3>
    """

    map_object.get_root().html.add_child(
        folium.Element(title_html)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    map_object.save(str(OUTPUT_FILE))

    print("\nHotspot map created successfully!")
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
