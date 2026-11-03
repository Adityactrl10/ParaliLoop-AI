
import pandas as pd
from pathlib import Path

# ==========================================
# PARALILOOP AI - DATA PREPROCESSING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    BASE_DIR / "data" / "raw" /
    "fire_archive_SV-C2_836588.csv"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "cleaned_fires.csv"


def load_data():
    """Load the original NASA FIRMS dataset."""

    print("\nLoading NASA FIRMS dataset...")

    if not RAW_FILE.exists():
        print("ERROR: NASA dataset not found!")
        return None

    data = pd.read_csv(RAW_FILE)

    print("Dataset loaded successfully!")
    print("Original records:", len(data))

    return data


def clean_data(data):
    """Clean and validate satellite fire detections."""

    print("\nStarting data cleaning...")

    original_count = len(data)

    # 1. Remove exact duplicate rows
    data = data.drop_duplicates().copy()

    # 2. Convert important columns to numeric values
    numeric_columns = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "frp"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    # 3. Remove records without valid location or date
    required_columns = [
        "latitude",
        "longitude",
        "acq_date"
    ]

    data = data.dropna(
        subset=required_columns
    ).copy()

    # 4. Keep only valid geographic coordinates
    data = data[
        data["latitude"].between(-90, 90) &
        data["longitude"].between(-180, 180)
    ].copy()

    # 5. Convert acquisition date
    data["acq_date"] = pd.to_datetime(
        data["acq_date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["acq_date"]
    ).copy()

    # 6. Create useful date features
    data["year"] = data["acq_date"].dt.year
    data["month"] = data["acq_date"].dt.month
    data["day"] = data["acq_date"].dt.day

    # 7. Sort detections by date
    data = data.sort_values(
        by="acq_date"
    ).reset_index(drop=True)

    print("\nCleaning completed!")

    print("Original records:", original_count)
    print("Clean records:", len(data))
    print("Removed records:", original_count - len(data))

    return data


def save_data(data):
    """Save cleaned data without changing the original file."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved successfully!")
    print("Saved location:", OUTPUT_FILE)


def main():

    data = load_data()

    if data is None:
        return

    cleaned_data = clean_data(data)

    save_data(cleaned_data)

    print("\n========== FINAL SUMMARY ==========")
    print("Total cleaned detections:", len(cleaned_data))

    print("\nFirst 5 cleaned records:")
    print(cleaned_data.head())

    print("\nMissing values:")
    print(cleaned_data.isnull().sum())

    print("\nParaliLoop AI preprocessing completed!")


if __name__ == "__main__":
    main()
