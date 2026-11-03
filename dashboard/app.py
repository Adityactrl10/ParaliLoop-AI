
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

from pathlib import Path
from datetime import date

# =====================================================
# PARALILOOP AI - PREMIUM AGRICULTURE INTELLIGENCE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MAP_DIR = BASE_DIR / "dashboard" / "maps"

MODEL_FILE = MODEL_DIR / "paraliloop_model.pkl"
METRICS_FILE = MODEL_DIR / "model_metrics.json"

st.set_page_config(
    page_title="ParaliLoop AI | Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PREMIUM DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --green: #176B45;
    --deep: #103D2B;
    --emerald: #27A66D;
    --lime: #B5D99C;
    --cream: #F6F8F2;
    --white: #FFFFFF;
    --muted: #748477;
}

.stApp {
    background: #F6F8F2;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: #103D2B;
}

section[data-testid="stSidebar"] * {
    color: #F2F7F1 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.05);
    padding: 11px 14px;
    border-radius: 9px;
    margin-bottom: 5px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.14);
}

h1, h2, h3 {
    color: #103D2B !important;
    font-family: 'Manrope', sans-serif;
}

p, label {
    color: #405348;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2EADF;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(20,70,45,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #607568 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #176B45 !important;
    font-weight: 800;
}

div[data-testid="stMetricDelta"] {
    color: #27A66D !important;
}

div.stButton > button {
    background: #176B45;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #103D2B;
    color: white;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #E1EADF;
}

.hero {
    background: linear-gradient(115deg, #103D2B, #176B45 65%, #258657);
    padding: 35px 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 28px rgba(16,61,43,0.15);
}

.hero h1 {
    color: white !important;
    font-size: 39px;
    margin: 0;
    font-weight: 800;
}

.hero p {
    color: #DDEDE1 !important;
    font-size: 15px;
}

.hero-tag {
    color: #D6F0DB;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}

.section-label {
    color: #258657;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.info-card {
    background: white;
    border: 1px solid #E2EADF;
    border-radius: 16px;
    padding: 22px;
    height: 100%;
}

.info-card h3 {
    margin-top: 8px;
    font-size: 18px;
}

.info-card p {
    color: #718076;
    font-size: 14px;
}

.status-pill {
    display: inline-block;
    background: #DDF2E4;
    color: #176B45;
    border-radius: 30px;
    padding: 6px 13px;
    font-size: 12px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #829087;
    padding: 22px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_csv(filename):

    path = DATA_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return None


@st.cache_resource
def load_model():

    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    return None


def load_metrics():

    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as file:
            return json.load(file)

    return None


fires = load_csv("cleaned_fires.csv")
monthly = load_csv("monthly_analysis.csv")
yearly = load_csv("yearly_analysis.csv")
hotspots = load_csv("fire_risk_zones.csv")
importance = load_csv("feature_importance.csv")

model = load_model()
metrics = load_metrics()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 25px 5px;">
        <div style="font-size:38px;">🌾</div>
        <div style="font-size:25px;font-weight:800;">
            PARALILOOP
        </div>
        <div style="font-size:11px;letter-spacing:2px;color:#B5D99C;">
            AI AGRICULTURE INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "Fire Hotspot Map",
            "Historical Analytics",
            "AI Intelligence",
            "Risk Prediction"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### PROJECT STATUS")

    st.markdown(
        '<span class="status-pill">● System Online</span>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("PILOT REGION")
    st.markdown("Punjab & Haryana")

    st.caption("DATA SOURCE")
    st.markdown("NASA FIRMS · VIIRS")

    st.caption("TEAM")
    st.markdown("TriVortex")

    st.divider()

    st.caption("Sustainability Hackathon · IIT Indore")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def show_hero(title, subtitle, tag="AGRICULTURE INTELLIGENCE"):

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">{tag}</div>
        <br>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def show_section(title, subtitle=None):

    st.markdown(
        f'<div class="section-label">{title.upper()}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.caption(subtitle)


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    show_hero(
        "See the fire. Understand the risk.",
        "Satellite-powered fire intelligence and machine learning "
        "for agricultural sustainability across Punjab and Haryana."
    )

    st.markdown("## Platform Overview")

    if fires is not None:

        total = len(fires)

        avg_frp = fires["frp"].mean()

        observation_days = fires["acq_date"].nunique()

        high_frp = len(
            fires[fires["frp"] >= fires["frp"].quantile(0.90)]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Satellite Detections",
            f"{total:,}"
        )

        c2.metric(
            "Average Fire Power",
            f"{avg_frp:.2f} MW"
        )

        c3.metric(
            "Observation Days",
            f"{observation_days:,}"
        )

        c4.metric(
            "Top 10% FRP Records",
            f"{high_frp:,}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    show_section(
        "Our Intelligence System",
        "Four capabilities powering the ParaliLoop platform."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">🛰️</div>
            <h3>Satellite Intelligence</h3>
            <p>Analyse real NASA VIIRS fire detections
            across the pilot region.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">📍</div>
            <h3>Hotspot Identification</h3>
            <p>Discover geographical areas with repeated
            satellite fire detections.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">🤖</div>
            <h3>Machine Learning</h3>
            <p>Estimate next-day satellite fire-detection
            likelihood using historical patterns.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div style="font-size:30px;">🌱</div>
            <h3>Sustainability Intelligence</h3>
            <p>Support future agricultural residue management
            and targeted interventions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    show_section("Historical Detection Trend")

    if monthly is not None:

        fig = px.area(
            monthly,
            x="month",
            y="fire_detections",
            markers=True,
            color_discrete_sequence=["#21966A"]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Detections",
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Scientific note: Satellite fire detections are not "
        "automatically confirmed agricultural residue-burning incidents."
    )


# =====================================================
# PAGE 2: FIRE HOTSPOT MAP
# =====================================================

elif page == "Fire Hotspot Map":

    show_hero(
        "Geospatial Fire Intelligence",
        "Explore historical satellite fire-detection concentrations.",
        "GEOSPATIAL ANALYTICS"
    )

    map_path = (
        MAP_DIR / "punjab_haryana_fire_map.html"
    )

    if map_path.exists():

        html = map_path.read_text(encoding="utf-8")

        st.components.v1.html(
            html,
            height=760,
            scrolling=True
        )

    else:

        st.warning(
            "Interactive map not found. "
            "Run src/create_interactive_map.py."
        )

    st.caption(
        "Map points represent aggregated satellite detections "
        "in approximately 0.1-degree grid cells."
    )


# =====================================================
# PAGE 3: HISTORICAL ANALYTICS
# =====================================================

elif page == "Historical Analytics":

    show_hero(
        "Historical Fire Analytics",
        "Understand when and where satellite fire detections occur.",
        "DATA EXPLORATION"
    )

    if monthly is not None:

        show_section("Monthly Distribution")

        fig = px.bar(
            monthly,
            x="month",
            y="fire_detections",
            color="fire_detections",
            color_continuous_scale=[
                "#D9F0DF",
                "#55B889",
                "#176B45"
            ]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

    if yearly is not None:

        show_section("Year-wise Distribution")

        fig = px.line(
            yearly,
            x="year",
            y="fire_detections",
            markers=True
        )

        fig.update_traces(
            line_color="#176B45",
            line_width=4,
            marker_size=10
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    if fires is not None:

        show_section("Fire Radiative Power")

        fig = px.histogram(
            fires,
            x="frp",
            nbins=60,
            color_discrete_sequence=["#27A66D"]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    if hotspots is not None:

        show_section("Top Historical Detection Hotspots")

        st.dataframe(
            hotspots.head(20),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# PAGE 4: AI INTELLIGENCE
# =====================================================

elif page == "AI Intelligence":

    show_hero(
        "Machine Learning Intelligence",
        "Performance and feature analysis of our Random Forest model.",
        "MODEL PERFORMANCE"
    )

    if metrics is not None:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

        c3.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

        c4, c5 = st.columns(2)

        c4.metric(
            "F1 Score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )

        if metrics.get("roc_auc") is not None:

            c5.metric(
                "ROC-AUC",
                f"{metrics['roc_auc'] * 100:.2f}%"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        show_section("Model Configuration")

        a, b = st.columns(2)

        a.markdown(f"""
        <div class="info-card">
            <h3>Training Period</h3>
            <h2>{metrics['training_years']}</h2>
        </div>
        """, unsafe_allow_html=True)

        b.markdown(f"""
        <div class="info-card">
            <h3>Testing Period</h3>
            <h2>{metrics['testing_year']}</h2>
        </div>
        """, unsafe_allow_html=True)

        if importance is not None:

            st.markdown("<br>", unsafe_allow_html=True)

            show_section("Feature Importance")

            fig = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[
                    "#B5D99C",
                    "#176B45"
                ]
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.info(
            "This model predicts next-day satellite fire detection, "
            "not confirmed stubble-burning incidents."
        )

    else:

        st.warning("Model metrics are not available.")


# =====================================================
# PAGE 5: RISK PREDICTION
# =====================================================

elif page == "Risk Prediction":

    show_hero(
        "AI Prediction Centre",
        "Estimate next-day satellite fire-detection likelihood "
        "for a selected geographical grid cell.",
        "PREDICTIVE INTELLIGENCE"
    )

    if model is None:

        st.error(
            "Model not found. Run src/train_model.py first."
        )

    else:

        st.markdown("### Enter Location & Historical Data")

        c1, c2 = st.columns(2)

        with c1:

            latitude = st.number_input(
                "Latitude",
                min_value=27.5,
                max_value=32.7,
                value=30.0,
                step=0.1
            )

            longitude = st.number_input(
                "Longitude",
                min_value=73.5,
                max_value=77.7,
                value=75.5,
                step=0.1
            )

            selected_date = st.date_input(
                "Date",
                value=date(2025, 10, 15)
            )

        with c2:

            previous_day = st.number_input(
                "Previous day detections",
                min_value=0,
                value=2
            )

            previous_3_days = st.number_input(
                "Previous 3-day detections",
                min_value=0,
                value=5
            )

            previous_7_days = st.number_input(
                "Previous 7-day detections",
                min_value=0,
                value=10
            )

        if st.button(
            "GENERATE PREDICTION",
            type="primary",
            use_container_width=True
        ):

            input_data = pd.DataFrame([{
                "latitude_grid": round(latitude, 1),
                "longitude_grid": round(longitude, 1),
                "month": selected_date.month,
                "day_of_year": selected_date.timetuple().tm_yday,
                "previous_day_count": previous_day,
                "previous_3_day_count": previous_3_days,
                "previous_7_day_count": previous_7_days
            }])

            probability = model.predict_proba(
                input_data
            )[0][1]

            st.markdown("<br>", unsafe_allow_html=True)

            st.success("Prediction generated successfully!")

            st.metric(
                "Next-Day Detection Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            st.caption(
                "This is a model-estimated probability for a "
                "satellite detection, not a confirmed fire or "
                "stubble-burning probability."
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <b>PARALILOOP AI</b> · Built by Team TriVortex<br>
    Sustainability Hackathon · IIT Indore<br>
    Powered by NASA FIRMS and Machine Learning
</div>
""", unsafe_allow_html=True)
