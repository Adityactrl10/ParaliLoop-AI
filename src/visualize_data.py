
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PARALILOOP AI - DATA VISUALIZATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "dashboard" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "cleaned_fires.csv"


def load_data():

    print("\nLoading cleaned NASA data...")

    if not INPUT_FILE.exists():
        print("ERROR: cleaned_fires.csv not found!")
        return None

    data = pd.read_csv(INPUT_FILE)

    print("Records loaded:", len(data))

    return data


def monthly_chart(data):

    monthly = (
        data.groupby("month")
        .size()
        .reindex(range(1, 13), fill_value=0)
    )

    plt.figure(figsize=(10, 5))
    monthly.plot(kind="bar")

    plt.title("Monthly NASA Fire Detections")
    plt.xlabel("Month")
    plt.ylabel("Number of Detections")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "monthly_fires.png")
    plt.close()

    print("Monthly graph saved.")


def yearly_chart(data):

    yearly = data.groupby("year").size()

    plt.figure(figsize=(9, 5))
    yearly.plot(kind="bar")

    plt.title("Year-wise NASA Fire Detections")
    plt.xlabel("Year")
    plt.ylabel("Number of Detections")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "yearly_fires.png")
    plt.close()

    print("Yearly graph saved.")


def location_chart(data):

    locations = (
        data.groupby(
            ["latitude_grid", "longitude_grid"]
        )
        .size()
        .reset_index(name="detections")
        .nlargest(20, "detections")
    )

    labels = (
        locations["latitude_grid"].astype(str)
        + ", "
        + locations["longitude_grid"].astype(str)
    )

    plt.figure(figsize=(11, 7))
    plt.barh(labels, locations["detections"])

    plt.title("Top 20 Fire Detection Grid Locations")
    plt.xlabel("Number of Detections")
    plt.ylabel("Latitude, Longitude")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "top_locations.png")
    plt.close()

    print("Location graph saved.")


def frp_chart(data):

    frp = data["frp"].dropna()
    frp = frp[frp >= 0]

    plt.figure(figsize=(9, 5))
    plt.hist(frp, bins=50)

    plt.title("Fire Radiative Power Distribution")
    plt.xlabel("FRP (MW)")
    plt.ylabel("Number of Detections")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "frp_distribution.png")
    plt.close()

    print("FRP graph saved.")


def main():

    data = load_data()

    if data is None:
        return

    monthly_chart(data)
    yearly_chart(data)

    # Create grid coordinates if they are not already present
    if "latitude_grid" not in data.columns:
        data["latitude_grid"] = (
            data["latitude"] * 10
        ).round() / 10

        data["longitude_grid"] = (
            data["longitude"] * 10
        ).round() / 10

    location_chart(data)
    frp_chart(data)

    print("\nAll visualizations created successfully!")
    print("Saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
