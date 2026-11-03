
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# PARALILOOP AI - INTERACTIVE FIRE MAP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_fires.csv"
OUTPUT_DIR = BASE_DIR / "dashboard" / "maps"
OUTPUT_FILE = OUTPUT_DIR / "punjab_haryana_fire_map.html"


def main():

    print("\nLoading NASA fire detections...")

    if not INPUT_FILE.exists():
        print("ERROR: Cleaned dataset not found!")
        print("Expected:", INPUT_FILE)
        return

    data = pd.read_csv(INPUT_FILE)

    required_columns = ["latitude", "longitude", "acq_date", "frp"]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        print("ERROR: Missing columns:", missing)
        return

    data = data.dropna(subset=required_columns).copy()

    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data["frp"] = pd.to_numeric(data["frp"], errors="coerce")
    data["acq_date"] = pd.to_datetime(data["acq_date"], errors="coerce")

    data = data.dropna(subset=required_columns)

    # Punjab and Haryana approximate geographical bounds
    data = data[
        data["latitude"].between(27.5, 32.7)
        & data["longitude"].between(73.5, 77.7)
    ].copy()

    # October-November seasonal detections
    seasonal = data[
        data["acq_date"].dt.month.isin([10, 11])
    ].copy()

    if seasonal.empty:
        print("No October-November detections found!")
        return

    # Aggregate detections into 0.1-degree grid cells
    seasonal["latitude_grid"] = (
        seasonal["latitude"] * 10
    ).round() / 10

    seasonal["longitude_grid"] = (
        seasonal["longitude"] * 10
    ).round() / 10

    hotspots = (
        seasonal.groupby(["latitude_grid", "longitude_grid"])
        .agg(
            detections=("latitude", "size"),
            average_frp=("frp", "mean")
        )
        .reset_index()
    )

    print("Seasonal detections:", len(seasonal))
    print("Grid cells:", len(hotspots))

    # Create interactive map
    fig = px.scatter_geo(
        hotspots,
        lat="latitude_grid",
        lon="longitude_grid",
        size="detections",
        color="detections",
        hover_name="detections",
        hover_data={
            "latitude_grid": ":.2f",
            "longitude_grid": ":.2f",
            "average_frp": ":.2f",
            "detections": True
        },
        color_continuous_scale="YlOrRd",
        size_max=24,
        projection="mercator",
        title="ParaliLoop AI | Punjab & Haryana Fire Hotspots"
    )

    # Correct geographical axis configuration
    fig.update_geos(
        lataxis=dict(
            range=[27.5, 32.7],
            showgrid=True,
            gridcolor="rgb(210, 220, 215)"
        ),
        lonaxis=dict(
            range=[73.5, 77.7],
            showgrid=True,
            gridcolor="rgb(210, 220, 215)"
        ),
        showland=True,
        showocean=True,
        showcountries=True,
        showcoastlines=True,
        showframe=True,
        landcolor="rgb(239, 243, 238)",
        oceancolor="rgb(224, 237, 245)",
        countrycolor="rgb(130, 145, 135)",
        coastlinecolor="rgb(130, 145, 135)",
        bgcolor="rgb(250, 252, 249)"
    )

    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=0.5, color="white")
        )
    )

    fig.update_layout(
        height=800,
        margin=dict(l=20, r=30, t=75, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            size=13,
            color="#183B32"
        ),
        title=dict(
            x=0.03,
            font=dict(size=22, color="#145A42")
        ),
        coloraxis_colorbar=dict(
            title="Detections",
            thickness=16
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        str(OUTPUT_FILE),
        include_plotlyjs=True,
        config={
            "responsive": True,
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"]
        }
    )

    print("\n====================================")
    print("INTERACTIVE MAP CREATED SUCCESSFULLY")
    print("====================================")
    print("Total seasonal detections:", len(seasonal))
    print("Grid cells displayed:", len(hotspots))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
