
import pandas as pd
from pathlib import Path

# ==========================================
# PARALILOOP AI - HOTSPOT ANALYSIS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "cleaned_fires.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "fire_risk_zones.csv"


def load_data():

    print("\nLoading cleaned NASA data...")

    if not INPUT_FILE.exists():
        print("ERROR: Cleaned dataset not found!")
        return None

    data = pd.read_csv(INPUT_FILE)
    data["acq_date"] = pd.to_datetime(
        data["acq_date"], errors="coerce"
    )

    data = data.dropna(
        subset=["latitude", "longitude", "acq_date"]
    )

    print("Records loaded:", len(data))

    return data


def identify_seasonal_detections(data):

    print("\nFiltering October-November detections...")

    seasonal = data[
        data["acq_date"].dt.month.isin([10, 11])
    ].copy()

    print("Seasonal detections:", len(seasonal))

    return seasonal


def calculate_hotspots(data):

    print("\nCalculating geographical hotspots...")

    # Group nearby detections into approximately 0.1-degree cells
    data["latitude_grid"] = (
        data["latitude"] * 10
    ).round() / 10

    data["longitude_grid"] = (
        data["longitude"] * 10
    ).round() / 10

    hotspots = (
        data.groupby(
            ["latitude_grid", "longitude_grid"]
        )
        .agg(
            total_detections=("latitude", "size"),
            active_days=("acq_date", "nunique"),
            average_frp=("frp", "mean"),
            years_observed=("year", "nunique")
        )
        .reset_index()
    )

    # Relative hotspot categories based on detection counts
    low_threshold = hotspots["total_detections"].quantile(0.50)
    high_threshold = hotspots["total_detections"].quantile(0.90)

    def classify(count):

        if count >= high_threshold:
            return "High"

        elif count >= low_threshold:
            return "Medium"

        return "Low"

    hotspots["relative_hotspot_level"] = (
        hotspots["total_detections"].apply(classify)
    )

    hotspots = hotspots.sort_values(
        by="total_detections",
        ascending=False
    )

    return hotspots


def main():

    data = load_data()

    if data is None:
        return

    seasonal = identify_seasonal_detections(data)

    if seasonal.empty:
        print("No October-November detections found.")
        return

    hotspots = calculate_hotspots(seasonal)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    hotspots.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== HOTSPOT RESULTS ==========")

    print("Total geographical grid cells:", len(hotspots))

    print("\nTop 15 hotspot locations:")

    print(
        hotspots.head(15).to_string(index=False)
    )

    print("\nHotspot category summary:")

    print(
        hotspots["relative_hotspot_level"]
        .value_counts()
        .to_string()
    )

    print("\nResults saved to:", OUTPUT_FILE)

    print("\nHotspot analysis completed!")


if __name__ == "__main__":
    main()
