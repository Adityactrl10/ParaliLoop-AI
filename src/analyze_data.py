
import pandas as pd
from pathlib import Path

# ==========================================
# PARALILOOP AI - FIRE DATA ANALYSIS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "cleaned_fires.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"


def load_data():

    print("\nLoading cleaned NASA data...")

    if not INPUT_FILE.exists():
        print("ERROR: Cleaned dataset not found!")
        return None

    data = pd.read_csv(INPUT_FILE)

    data["acq_date"] = pd.to_datetime(data["acq_date"])

    print("Records loaded:", len(data))

    return data


def analyze_monthly(data):

    print("\n========== MONTHLY ANALYSIS ==========")

    monthly = (
        data.groupby("month")
        .size()
        .reset_index(name="fire_detections")
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_analysis.csv",
        index=False
    )

    print(monthly.to_string(index=False))

    return monthly


def analyze_yearly(data):

    print("\n========== YEARLY ANALYSIS ==========")

    yearly = (
        data.groupby("year")
        .size()
        .reset_index(name="fire_detections")
    )

    yearly.to_csv(
        OUTPUT_DIR / "yearly_analysis.csv",
        index=False
    )

    print(yearly.to_string(index=False))

    return yearly


def analyze_locations(data):

    print("\n========== TOP DETECTION LOCATIONS ==========")

    # Group coordinates into approximately 0.1-degree grid cells
    data = data.copy()

    data["latitude_grid"] = (
        data["latitude"] * 10
    ).round() / 10

    data["longitude_grid"] = (
        data["longitude"] * 10
    ).round() / 10

    locations = (
        data.groupby(["latitude_grid", "longitude_grid"])
        .size()
        .reset_index(name="fire_detections")
        .sort_values(
            by="fire_detections",
            ascending=False
        )
        .head(20)
    )

    locations.to_csv(
        OUTPUT_DIR / "top_fire_locations.csv",
        index=False
    )

    print(locations.to_string(index=False))

    return locations


def analyze_frp(data):

    print("\n========== FIRE RADIATIVE POWER ==========")

    print(data["frp"].describe())

    print(
        "\nTotal recorded FRP:",
        round(data["frp"].sum(), 2)
    )

    print(
        "Average FRP:",
        round(data["frp"].mean(), 2)
    )


def main():

    data = load_data()

    if data is None:
        return

    analyze_monthly(data)
    analyze_yearly(data)
    analyze_locations(data)
    analyze_frp(data)

    print("\nAnalysis files saved in data/processed/")
    print("\nParaliLoop AI analysis completed!")


if __name__ == "__main__":
    main()
