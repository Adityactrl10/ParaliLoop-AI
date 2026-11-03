
import geopandas as gpd
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent
BOUNDARY_DIR = BASE_DIR / "data" / "boundaries"

districts = gpd.read_file(BOUNDARY_DIR / "india_districts.geojson")
states = gpd.read_file(BOUNDARY_DIR / "india_states.geojson")

# Ensure both datasets use the same coordinate system
districts = districts.to_crs("EPSG:4326")
states = states.to_crs("EPSG:4326")

# Match using the largest area of intersection
districts["state_name"] = "Unknown"

for index, district in districts.iterrows():
    overlaps = states.geometry.intersection(district.geometry).area
    districts.at[index, "state_name"] = states.loc[overlaps.idxmax(), "shapeName"]

# Normalize state names to handle accented characters
def normalize_name(name):
    return unicodedata.normalize("NFKD", str(name)).encode(
        "ascii", "ignore"
    ).decode().lower().strip()

# Select Punjab and Haryana
selected = districts[
    districts["state_name"].apply(normalize_name).isin(
        ["punjab", "haryana"]
    )
].copy()

output = BOUNDARY_DIR / "punjab_haryana_districts.geojson"
selected.to_file(output, driver="GeoJSON")

print("Matching completed!")
print("Total selected districts:", len(selected))
print("\nDistricts by state:")
print(selected.groupby("state_name")["shapeName"].count())
print("\nDistrict names:")
print(selected[["shapeName", "state_name"]].to_string(index=False))
print("\nSaved:", output)