import json
import unicodedata
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.element import Element

# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "processed" / "cleaned_fires.csv"
BOUNDARIES = BASE / "data" / "boundaries" / "punjab_haryana_districts.geojson"
OUTPUT = BASE / "dashboard" / "maps" / "punjab_haryana_fire_map.html"

GRID_SIZE = 0.05
MAX_INDIVIDUAL_POINTS = 100000

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading fire data and district boundaries...")

if not DATA.exists():
    raise FileNotFoundError(f"Fire data not found: {DATA}")

if not BOUNDARIES.exists():
    raise FileNotFoundError(f"District boundary file not found: {BOUNDARIES}")

fires = pd.read_csv(DATA)
districts = gpd.read_file(BOUNDARIES)

# Standardize CSV column names
fires.columns = [str(c).strip().lower() for c in fires.columns]
districts.columns = [str(c).strip() for c in districts.columns]

required_fire = {
    "latitude", "longitude", "frp", "year", "confidence"
}

missing_fire = required_fire - set(fires.columns)

if missing_fire:
    raise ValueError(
        f"Missing columns in cleaned_fires.csv: {sorted(missing_fire)}. "
        f"Available columns: {list(fires.columns)}"
    )

required_boundary = {
    "shapeName", "state_name", "geometry"
}

missing_boundary = required_boundary - set(districts.columns)

if missing_boundary:
    raise ValueError(
        f"Missing columns in district GeoJSON: {sorted(missing_boundary)}. "
        f"Available columns: {list(districts.columns)}"
    )

# ============================================================
# 3. CLEAN AND FILTER DISTRICT BOUNDARIES
# ============================================================

def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value))
    return value.encode("ascii", "ignore").decode().strip().lower()


districts = districts.to_crs("EPSG:4326").copy()

districts["state_key"] = districts["state_name"].apply(normalize_name)

districts = districts[
    districts["state_key"].isin(["punjab", "haryana"])
].copy()

districts["state_name"] = districts["state_key"].map({
    "punjab": "Punjab",
    "haryana": "Haryana"
})

if districts.empty:
    raise RuntimeError(
        "No Punjab/Haryana districts found in the boundary file."
    )

# ============================================================
# 4. CLEAN FIRE DETECTIONS
# ============================================================

for column in ["latitude", "longitude", "frp", "year"]:
    fires[column] = pd.to_numeric(
        fires[column],
        errors="coerce"
    )

fires = fires.dropna(
    subset=["latitude", "longitude", "year"]
).copy()

fires = fires[
    fires["latitude"].between(26, 33)
    & fires["longitude"].between(72, 79)
].copy()

fires["frp"] = fires["frp"].fillna(0)

fires["year"] = fires["year"].astype(int).astype(str)


def classify_confidence(value):

    value = str(value).strip().lower()

    if value in {"h", "high"}:
        return "High"

    if value in {"n", "nominal", "medium"}:
        return "Nominal"

    if value in {"l", "low"}:
        return "Low"

    try:
        number = float(value)

        if number >= 80:
            return "High"

        if number >= 30:
            return "Nominal"

        return "Low"

    except (TypeError, ValueError):
        return "Unknown"


fires["confidence_class"] = fires["confidence"].apply(
    classify_confidence
)

if fires.empty:
    raise RuntimeError(
        "No valid fire detections remain after cleaning coordinates."
    )

# ============================================================
# 5. SPATIAL JOIN - KEEP FIRE AND DISTRICT COLUMNS SEPARATE
# ============================================================

# Keep only required fire columns.
# This avoids duplicate state_name columns during spatial join.

fire_map_data = fires[
    [
        "latitude",
        "longitude",
        "frp",
        "year",
        "confidence_class"
    ]
].copy()

fire_points = gpd.GeoDataFrame(
    fire_map_data,
    geometry=gpd.points_from_xy(
        fire_map_data["longitude"],
        fire_map_data["latitude"]
    ),
    crs="EPSG:4326"
)

# Prepare district data with a unique district name column.

districts_for_join = districts[
    ["shapeName", "state_name", "geometry"]
].copy()

districts_for_join = districts_for_join.rename(
    columns={
        "shapeName": "district_name"
    }
)

# Spatial join

points = gpd.sjoin(
    fire_points,
    districts_for_join,
    how="inner",
    predicate="within"
)

points = points.drop(
    columns=["index_right"],
    errors="ignore"
)

# Verify all required columns exist.

required_point_columns = {
    "latitude",
    "longitude",
    "frp",
    "year",
    "confidence_class",
    "district_name",
    "state_name",
    "geometry"
}

missing_points = required_point_columns - set(points.columns)

if missing_points:
    raise ValueError(
        f"Spatial join did not produce required columns: "
        f"{sorted(missing_points)}. "
        f"Available columns: {list(points.columns)}"
    )

if points.empty:
    raise RuntimeError(
        "No fire detections matched Punjab/Haryana districts. "
        "Check that the boundary GeoJSON and coordinates use EPSG:4326."
    )

points["year"] = points["year"].astype(int).astype(str)
points["state_name"] = points["state_name"].astype(str)
points["confidence_class"] = points["confidence_class"].astype(str)

print("Matched detections:", f"{len(points):,}")

# ============================================================
# 6. BUILD FILTERABLE GRID AND INDIVIDUAL-POINT SAMPLE
# ============================================================

points["grid_lat"] = (
    (points["latitude"] / GRID_SIZE).round() * GRID_SIZE
).round(4)

points["grid_lon"] = (
    (points["longitude"] / GRID_SIZE).round() * GRID_SIZE
).round(4)

grid = (
    points.groupby(
        [
            "grid_lat",
            "grid_lon",
            "state_name",
            "year",
            "confidence_class"
        ],
        as_index=False
    )
    .agg(
        count=("latitude", "size"),
        mean_frp=("frp", "mean"),
        max_frp=("frp", "max")
    )
)

grid_data = grid.to_dict(orient="records")

for item in grid_data:

    item["grid_lat"] = float(item["grid_lat"])
    item["grid_lon"] = float(item["grid_lon"])
    item["count"] = int(item["count"])
    item["mean_frp"] = float(round(item["mean_frp"], 2))
    item["max_frp"] = float(round(item["max_frp"], 2))
    item["year"] = str(item["year"])

# Representative sample for individual-point zoom view.
# Complete data is still used for all KPIs and counts.

if len(points) > MAX_INDIVIDUAL_POINTS:

    individual_points = points.sample(
        n=MAX_INDIVIDUAL_POINTS,
        random_state=42
    ).copy()

else:

    individual_points = points.copy()

individual_data = individual_points[
    [
        "latitude",
        "longitude",
        "frp",
        "year",
        "state_name",
        "confidence_class"
    ]
].to_dict(orient="records")

for item in individual_data:

    item["latitude"] = float(item["latitude"])
    item["longitude"] = float(item["longitude"])
    item["frp"] = float(item["frp"])
    item["year"] = str(item["year"])

years = sorted(points["year"].unique().tolist())

total_detections = int(len(points))

print("Hotspot grid records:", f"{len(grid):,}")
print("Individual points in HTML sample:", f"{len(individual_data):,}")
print("Districts:", len(districts))

# ============================================================
# 7. ADD FILTERABLE STATISTICS TO DISTRICTS
# ============================================================

stats = (
    points.groupby(
        [
            "district_name",
            "state_name",
            "year",
            "confidence_class"
        ]
    )
    .size()
    .to_dict()
)

district_features = json.loads(districts.to_json())

for feature in district_features["features"]:

    properties = feature["properties"]

    district_name = properties["shapeName"]
    state_name = properties["state_name"]

    district_stats = {}

    for (name, state, year, confidence), count in stats.items():

        if name == district_name and state == state_name:

            district_stats[f"{year}|{confidence}"] = int(count)

    properties["fire_stats"] = district_stats

# ============================================================
# 8. CREATE BASE MAP AND DISTRICT LAYER
# ============================================================

print("Creating interactive map...")

m = folium.Map(
    location=[30.5, 75.5],
    zoom_start=7,
    tiles=None,
    control_scale=True,
    prefer_canvas=True
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Tiles © Esri",
    name="Esri World Street Map",
    overlay=False,
    control=True
).add_to(m)

district_layer = folium.GeoJson(
    district_features,
    name="District Boundaries",
    style_function=lambda feature: {
        "fillColor": "#DDEBDD",
        "color": "#174A36",
        "weight": 1.2,
        "fillOpacity": 0.10
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["shapeName", "state_name"],
        aliases=["District:", "State:"],
        sticky=True
    )
).add_to(m)

minx, miny, maxx, maxy = districts.total_bounds

m.fit_bounds([
    [miny, minx],
    [maxy, maxx]
])
# ============================================================
# 9. TITLE, FILTER PANEL, AND LEGEND
# ============================================================

title_html = """
<div style="
position:fixed;
top:12px;
left:50%;
transform:translateX(-50%);
z-index:9999;
background:white;
padding:12px 25px;
border-radius:12px;
box-shadow:0 2px 10px #0003;
text-align:center;
font-family:Arial;
pointer-events:none;
">
    <div style="font-size:21px;font-weight:bold;color:#174A36;">
        ParaliLoop AI
    </div>
    <div style="font-size:12px;color:#555;">
        Punjab &amp; Haryana Fire Intelligence
    </div>
</div>
"""

m.get_root().html.add_child(Element(title_html))

year_options = "".join(
    f'<option value="{year}">{year}</option>'
    for year in years
)

panel_html = f"""
<div id="pl-panel" style="
position:fixed;
top:100px;
left:12px;
z-index:9999;
width:245px;
background:white;
padding:14px;
border-radius:12px;
box-shadow:0 2px 12px #0003;
font-family:Arial;
color:#333;
">

<h3 style="margin:0 0 12px;color:#174A36;">
Fire Intelligence
</h3>

<label>Year</label>

<select id="pl-year"
style="width:100%;padding:7px;margin:5px 0 10px;">

<option value="All">All years</option>
{year_options}

</select>

<label>State</label>

<select id="pl-state"
style="width:100%;padding:7px;margin:5px 0 10px;">

<option value="All">Punjab + Haryana</option>
<option value="Punjab">Punjab</option>
<option value="Haryana">Haryana</option>

</select>

<label>Confidence</label>

<select id="pl-confidence"
style="width:100%;padding:7px;margin:5px 0 12px;">

<option value="All">All confidence levels</option>
<option value="High">High</option>
<option value="Nominal">Nominal</option>
<option value="Low">Low</option>
<option value="Unknown">Unknown</option>

</select>

<hr>

<div style="font-size:12px;color:#666;">
Filtered detections
</div>

<div id="pl-total"
style="font-size:25px;font-weight:bold;color:#174A36;">
{total_detections:,}
</div>

<div style="font-size:12px;color:#666;margin-top:8px;">
High-confidence detections
</div>

<div id="pl-high"
style="font-size:19px;font-weight:bold;">
—
</div>

<div style="font-size:12px;color:#666;margin-top:8px;">
Average FRP (MW)
</div>

<div id="pl-frp"
style="font-size:19px;font-weight:bold;">
—
</div>

<button id="pl-reset" style="
width:100%;
margin-top:13px;
padding:9px;
background:#174A36;
color:white;
border:0;
border-radius:6px;
cursor:pointer;
">
Reset filters
</button>

</div>

<div style="
position:fixed;
bottom:25px;
right:15px;
z-index:9999;
background:white;
padding:13px;
border-radius:10px;
box-shadow:0 2px 10px #0003;
font:12px Arial;
width:220px;
">

<b style="color:#174A36;">Map Legend</b>

<p>🔴 High fire concentration</p>
<p>🟠 Medium fire concentration</p>
<p>🟡 Lower fire concentration</p>
<p>━ District boundaries</p>

<hr>

<span style="color:#666;">
Zoom out: aggregated grid cells.
Zoom in (13+): sampled individual satellite detections.
Satellite detections are not automatically confirmed
stubble-burning incidents.
</span>

</div>
"""

m.get_root().html.add_child(Element(panel_html))

# ============================================================
# 10. ZOOM-ADAPTIVE MAP AND FILTER LOGIC
# ============================================================

map_name = m.get_name()
district_layer_name = district_layer.get_name()

grid_json = json.dumps(
    grid_data,
    separators=(",", ":")
)

individual_json = json.dumps(
    individual_data,
    separators=(",", ":")
)

javascript = r"""
window.addEventListener('load', function() {

    const map = __MAP_NAME__;
    const districtLayer = __DISTRICT_NAME__;
    const gridData = __GRID_DATA__;
    const individualData = __INDIVIDUAL_DATA__;

    const aggregateLayer = L.layerGroup().addTo(map);
    const individualLayer = L.layerGroup().addTo(map);

    const yearInput = document.getElementById('pl-year');
    const stateInput = document.getElementById('pl-state');
    const confidenceInput = document.getElementById('pl-confidence');

    function matches(item, year, state, confidence) {

        return (
            (year === 'All' || String(item.year) === year) &&
            (state === 'All' || item.state_name === state) &&
            (confidence === 'All' ||
             item.confidence_class === confidence)
        );
    }

    function confidenceColor(value) {

        if (value === 'High') return '#E63946';

        if (value === 'Nominal') return '#F28C28';

        return '#E9C46A';
    }

    function updateMap() {

        aggregateLayer.clearLayers();
        individualLayer.clearLayers();

        const year = yearInput.value;
        const state = stateInput.value;
        const confidence = confidenceInput.value;

        const zoom = map.getZoom();

        let total = 0;
        let high = 0;
        let weightedFrp = 0;

        const cells = new Map();

        // Aggregate matching records by grid cell.

        gridData.forEach(function(item) {

            if (!matches(item, year, state, confidence)) {
                return;
            }

            total += item.count;

            if (item.confidence_class === 'High') {
                high += item.count;
            }

            weightedFrp += item.mean_frp * item.count;

            const key =
                item.grid_lat.toFixed(4) + '|' +
                item.grid_lon.toFixed(4) + '|' +
                item.state_name;

            if (!cells.has(key)) {

                cells.set(key, {
                    lat: item.grid_lat,
                    lon: item.grid_lon,
                    state: item.state_name,
                    count: 0,
                    frpTotal: 0,
                    maxFrp: 0,
                    years: new Set()
                });
            }

            const cell = cells.get(key);

            cell.count += item.count;

            cell.frpTotal += item.mean_frp * item.count;

            cell.maxFrp = Math.max(
                cell.maxFrp,
                item.max_frp
            );

            cell.years.add(String(item.year));
        });

        // Zoom below 13: show aggregated grid cells.

        if (zoom < 13) {

            cells.forEach(function(cell) {

                const averageFrp = cell.count
                    ? cell.frpTotal / cell.count
                    : 0;

                const color = cell.count >= 100
                    ? '#E63946'
                    : cell.count >= 30
                    ? '#F28C28'
                    : '#E9C46A';

                const radius = zoom <= 9
                    ? Math.min(
                        8,
                        2.5 + Math.sqrt(cell.count) * 0.18
                    )
                    : Math.min(
                        6,
                        2 + Math.sqrt(cell.count) * 0.12
                    );

                L.circleMarker(
                    [cell.lat, cell.lon],
                    {
                        radius: radius,
                        color: color,
                        fillColor: color,
                        fillOpacity: zoom <= 9 ? 0.42 : 0.58,
                        weight: 0.7
                    }
                )
                .bindTooltip(
                    '<b>State:</b> ' + cell.state +
                    '<br><b>Grouped detections:</b> ' +
                    cell.count.toLocaleString('en-IN') +
                    '<br><b>Average FRP:</b> ' +
                    averageFrp.toFixed(2) + ' MW' +
                    '<br><b>Maximum FRP:</b> ' +
                    cell.maxFrp.toFixed(2) + ' MW' +
                    '<br><b>Years:</b> ' +
                    Array.from(cell.years).sort().join(', ')
                )
                .addTo(aggregateLayer);
            });

        } else {

            // Zoom 13+: show sampled individual detections
            // inside the visible map area.

            const bounds = map.getBounds();

            individualData.forEach(function(item) {

                if (!matches(item, year, state, confidence)) {
                    return;
                }

                const location = [
                    item.latitude,
                    item.longitude
                ];

                if (!bounds.contains(location)) {
                    return;
                }

                const color = confidenceColor(
                    item.confidence_class
                );

                L.circleMarker(
                    location,
                    {
                        radius: 3,
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.78,
                        weight: 0.6
                    }
                )
                .bindTooltip(
                    '<b>Satellite detection (sample)</b>' +
                    '<br>State: ' + item.state_name +
                    '<br>Year: ' + item.year +
                    '<br>Confidence: ' + item.confidence_class +
                    '<br>FRP: ' +
                    Number(item.frp).toFixed(2) + ' MW'
                )
                .addTo(individualLayer);
            });
        }

        // KPIs use the complete filtered dataset,
        // not the individual-point sample.

        document.getElementById('pl-total').textContent =
            total.toLocaleString('en-IN');

        document.getElementById('pl-high').textContent =
            high.toLocaleString('en-IN');

        document.getElementById('pl-frp').textContent =
            total > 0
                ? (weightedFrp / total).toFixed(2)
                : '—';

        // Update district shading.

        districtLayer.eachLayer(function(layer) {

            const properties = layer.feature.properties;

            let count = 0;

            if (
                state !== 'All' &&
                properties.state_name !== state
            ) {

                layer.setStyle({
                    fillOpacity: 0.02,
                    color: '#aaaaaa',
                    weight: 0.8
                });

                return;
            }

            Object.entries(
                properties.fire_stats || {}
            ).forEach(function(entry) {

                const parts = entry[0].split('|');

                if (
                    (year === 'All' || parts[0] === year) &&
                    (confidence === 'All' ||
                     parts[1] === confidence)
                ) {
                    count += entry[1];
                }
            });

            const fill = count >= 10000
                ? '#E76F51'
                : count >= 3000
                ? '#F4A261'
                : count > 0
                ? '#E9C46A'
                : '#DDE8DD';

            layer.setStyle({
                fillColor: fill,
                fillOpacity: count > 0 ? 0.25 : 0.07,
                color: '#174A36',
                weight: 1.2
            });
        });
    }

    // Filter event listeners.

    [
        yearInput,
        stateInput,
        confidenceInput
    ].forEach(function(input) {

        input.addEventListener('change', updateMap);
    });

    // Refresh map when zooming or moving.

    map.on('zoomend moveend', updateMap);

    // Reset all filters.

    document.getElementById('pl-reset').addEventListener(
        'click',
        function() {

            yearInput.value = 'All';
            stateInput.value = 'All';
            confidenceInput.value = 'All';

            updateMap();
        }
    );

    // Initial map render.

    updateMap();
});
"""

javascript = javascript.replace(
    "__MAP_NAME__",
    map_name
)

javascript = javascript.replace(
    "__DISTRICT_NAME__",
    district_layer_name
)

javascript = javascript.replace(
    "__GRID_DATA__",
    grid_json
)

javascript = javascript.replace(
    "__INDIVIDUAL_DATA__",
    individual_json
)

m.get_root().script.add_child(Element(javascript))

# ============================================================
# 11. SAVE MAP
# ============================================================

folium.LayerControl(collapsed=True).add_to(m)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

m.save(OUTPUT)

print()

print("ParaliLoop AI district fire map created successfully!")

print("Total mapped detections:", f"{total_detections:,}")

print("Years available:", ", ".join(years))

print("Districts:", len(districts))

print("Hotspot grid records:", f"{len(grid):,}")

print(
    "Individual points included in HTML sample:",
    f"{len(individual_data):,}"
)

print("Saved at:", OUTPUT)