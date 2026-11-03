
import pandas as pd
from pathlib import Path

# ==========================================
# PARALILOOP AI - PREDICTION DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "cleaned_fires.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "prediction_dataset.csv"


def load_data():

    print("\nLoading NASA data...")

    if not INPUT_FILE.exists():
        print("ERROR: Cleaned dataset not found!")
        return None

    data = pd.read_csv(INPUT_FILE)

    data["acq_date"] = pd.to_datetime(
        data["acq_date"], errors="coerce"
    )

    data = data.dropna(
        subset=["latitude", "longitude", "acq_date"]
    ).copy()

    data = data[
        data["acq_date"].dt.month.isin([10, 11])
    ].copy()

    data["latitude_grid"] = (
        data["latitude"] * 10
    ).round() / 10

    data["longitude_grid"] = (
        data["longitude"] * 10
    ).round() / 10

    print("Seasonal detections:", len(data))

    return data


def create_daily_dataset(data):

    print("\nCreating daily records...")

    daily = (
        data.groupby([
            "year",
            "latitude_grid",
            "longitude_grid",
            "acq_date"
        ])
        .size()
        .reset_index(name="detection_count")
    )

    calendars = []

    for year in sorted(daily["year"].unique()):

        year_data = daily[daily["year"] == year]

        cells = year_data[
            ["latitude_grid", "longitude_grid"]
        ].drop_duplicates()

        dates = pd.date_range(
            f"{year}-10-01",
            f"{year}-11-30",
            freq="D"
        )

        for cell in cells.itertuples(index=False):

            calendars.append(
                pd.DataFrame({
                    "year": year,
                    "latitude_grid": cell.latitude_grid,
                    "longitude_grid": cell.longitude_grid,
                    "acq_date": dates
                })
            )

    panel = pd.concat(
        calendars,
        ignore_index=True
    )

    panel = panel.merge(
        daily,
        on=[
            "year",
            "latitude_grid",
            "longitude_grid",
            "acq_date"
        ],
        how="left"
    )

    panel["detection_count"] = (
        panel["detection_count"].fillna(0)
    )

    panel = panel.sort_values([
        "year",
        "latitude_grid",
        "longitude_grid",
        "acq_date"
    ]).reset_index(drop=True)

    return panel


def create_features(panel):

    print("\nCreating prediction features...")

    group_columns = [
        "year",
        "latitude_grid",
        "longitude_grid"
    ]

    grouped = panel.groupby(group_columns)

    panel["previous_day_count"] = (
        grouped["detection_count"].shift(1)
    )

    panel["previous_3_day_count"] = (
        grouped["detection_count"]
        .transform(
            lambda s: s.shift(1).rolling(3).sum()
        )
    )

    panel["previous_7_day_count"] = (
        grouped["detection_count"]
        .transform(
            lambda s: s.shift(1).rolling(7).sum()
        )
    )

    panel["month"] = panel["acq_date"].dt.month
    panel["day_of_year"] = panel["acq_date"].dt.dayofyear

    panel["target_next_day"] = (
        grouped["detection_count"]
        .shift(-1)
        .gt(0)
        .astype(int)
    )

    # Remove first 7 days and last day of each season
    # because historical features or next-day labels
    # are not available for those dates.
    panel["day_of_season"] = (
        panel["acq_date"].dt.dayofyear
        - pd.to_datetime(
            panel["year"].astype(str) + "-10-01"
        ).dt.dayofyear
    )

    panel = panel[
        (panel["day_of_season"] >= 7) &
        (panel["acq_date"].dt.strftime("%m-%d") != "11-30")
    ].copy()

    feature_columns = [
        "previous_day_count",
        "previous_3_day_count",
        "previous_7_day_count"
    ]

    panel = panel.dropna(
        subset=feature_columns
    ).copy()

    panel[feature_columns] = (
        panel[feature_columns].astype(int)
    )

    return panel


def main():

    data = load_data()

    if data is None or data.empty:
        print("No seasonal data available.")
        return

    panel = create_daily_dataset(data)

    prediction_data = create_features(panel)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== FINAL SUMMARY ==========")

    print("Prediction records:", len(prediction_data))

    print("\nTarget distribution:")
    print(
        prediction_data["target_next_day"]
        .value_counts().to_string()
    )

    print("\nSaved:", OUTPUT_FILE)

    print("\nPrediction dataset rebuilt successfully!")


if __name__ == "__main__":
    main()
