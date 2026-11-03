import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
from pathlib import Path
from datetime import date

# =====================================================
# PARALILOOP AI - PROFESSIONAL DASHBOARD
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ai_agent import ask_gemini
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MAP_DIR = BASE_DIR / "dashboard" / "maps"

MODEL_FILE = MODEL_DIR / "paraliloop_model.pkl"
METRICS_FILE = MODEL_DIR / "model_metrics.json"

st.set_page_config(
    page_title="ParaliLoop AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROFESSIONAL DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F5F7F2;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 100%;
    padding: 2rem 2.5rem 3rem;
}

/* Hide Streamlit toolbar */



/* Sidebar */

section[data-testid="stSidebar"] {
    background: #103D2B;
}

section[data-testid="stSidebar"] * {
    color: #F2F7F1;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 5px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #285640;
}

/* Typography */

h1, h2, h3 {
    color: #103D2B !important;
}

p {
    color: #52665A;
}

hr {
    border-color: #E0E8DE;
}

/* Hero */

.hero {
    background: linear-gradient(110deg, #103D2B, #176B45);
    padding: 34px;
    border-radius: 18px;
    color: white;
    margin-bottom: 30px;
}

.hero-label {
    color: #BFE5C8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: white !important;
    font-size: clamp(28px, 3vw, 40px);
    font-weight: 800;
    margin: 15px 0;
}

.hero-description {
    color: #DCECE0 !important;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

/* KPI cards */

.kpi-card {
    background: white;
    border: 1px solid #E2EADF;
    border-radius: 15px;
    padding: 20px 17px;
    min-height: 140px;
    box-shadow: 0 4px 15px rgba(16,61,43,0.05);
}

.kpi-label {
    color: #52665A;
    font-size: 12px;
    font-weight: 700;
    line-height: 1.4;
    min-height: 34px;
}

.kpi-value {
    color: #176B45;
    font-size: clamp(20px, 2vw, 29px);
    font-weight: 800;
    margin-top: 9px;
    white-space: nowrap;
}

.kpi-note {
    color: #829087;
    font-size: 11px;
    margin-top: 7px;
}

/* Capability cards */

.panel {
    background: white;
    border: 1px solid #E2EADF;
    border-radius: 15px;
    padding: 24px;
    min-height: 185px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(16,61,43,0.04);
}

.panel-title {
    color: #103D2B;
    font-size: 17px;
    font-weight: 800;
    line-height: 1.5;
    margin-top: 12px;
}

.panel-description {
    color: #64776B;
    font-size: 14px;
    line-height: 1.7;
}

/* Buttons */
div.stButton > button {
    background: #176B45 !important;
    color: #FFFFFF !important;
    border: 1px solid #176B45 !important;
    border-radius: 10px;
    font-weight: 700 !important;
    min-height: 48px;
    opacity: 1 !important;
}

div.stButton > button p,
div.stButton > button span,
div.stButton > button div {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

div.stButton > button:hover {
    background: #103D2B !important;
    color: #FFFFFF !important;
    border-color: #103D2B !important;
}

/* Data tables */

div[data-testid="stDataFrame"] {
    border: 1px solid #E0E8DE;
    border-radius: 12px;
    overflow: hidden;
}

/* Footer */

.footer {
    text-align: center;
    color: #718176;
    font-size: 12px;
    line-height: 1.8;
    padding: 25px 0;
}

/* Match Streamlit header with dashboard background */

header[data-testid="stHeader"] {
    background-color: #F5F7F2 !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

/* Mobile responsive improvements */
@media (max-width: 768px) {

    .block-container {
        padding: 1rem 1rem 2rem !important;
    }

    .hero {
        padding: 24px 20px !important;
        border-radius: 16px !important;
    }

    .hero-title {
        font-size: 27px !important;
        line-height: 1.35 !important;
    }

    .hero-description {
        font-size: 14px !important;
    }

    .kpi-card {
        padding: 17px 14px !important;
        min-height: 125px !important;
    }

    .kpi-value {
        font-size: 22px !important;
        white-space: normal !important;
    }

    .panel {
        padding: 18px !important;
    }

    /* Improve suggested question buttons */
    div.stButton > button {
        min-height: 55px !important;
        white-space: normal !important;
        font-size: 13px !important;
        padding: 10px !important;
    }

    /* Improve sidebar branding visibility */
    section[data-testid="stSidebar"] {
        min-width: 250px;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* Prevent charts from overflowing */
    div[data-testid="stPlotlyChart"] {
        max-width: 100%;
        overflow-x: auto;
    }
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


@st.cache_data
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
# HELPER FUNCTIONS
# =====================================================

def show_hero(title, description, label):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">{label}</div>
        <div class="hero-title">{title}</div>
        <p class="hero-description">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def show_kpi(label, value, note):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans",
            color="#344B3C",
            size=13
        ),
        title=dict(
            font=dict(
                family="DM Sans",
                size=19,
                color="#103D2B"
            ),
            x=0.02
        ),
        margin=dict(l=25, r=25, t=65, b=35),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#DCE5DC",
            tickfont=dict(color="#52665A", size=12),
            title_font=dict(color="#344B3C")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E9EFE7",
            zeroline=False,
            tickfont=dict(color="#52665A", size=12),
            title_font=dict(color="#344B3C")
        ),
        legend=dict(
            font=dict(color="#344B3C")
        )
    )
    return fig


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 20px;">
        <div style="font-size:34px;">🌾</div>
        <div style="font-size:23px;font-weight:800;letter-spacing:1px;">
            PARALILOOP
        </div>
        <div style="font-size:10px;letter-spacing:2px;color:#B5D99C;">
            AI INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.caption("WORKSPACE")

    page = st.radio(
    "Navigation",
    [
        "Executive Overview",
        "Geospatial Intelligence",
        "Historical Analytics",
        "AI Model Insights",
        "Prediction Centre",
        "AI Farming Assistant"
    ],
    label_visibility="collapsed"
)

    st.divider()

    st.caption("PILOT REGION")
    st.markdown("Punjab & Haryana")

    st.caption("SATELLITE SOURCE")
    st.markdown("NASA FIRMS · VIIRS")

    st.caption("DEVELOPED BY")
    st.markdown("Team TriVortex")

    st.divider()

    st.markdown("""
    <div style="background:#1D563A;padding:12px;border-radius:10px;">
        <div style="font-size:11px;color:#BFE5C8;">
            PLATFORM STATUS
        </div>
        <div style="font-size:14px;font-weight:700;">
            ● Operational
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Sustainability Hackathon · IIT Indore")


# =====================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =====================================================

if page == "Executive Overview":

    show_hero(
        "Agricultural Fire Intelligence",
        "Satellite observations and machine learning for "
        "historical fire analysis across Punjab and Haryana.",
        "EXECUTIVE DASHBOARD"
    )

    st.markdown("### Platform Performance")

    if fires is not None:

        total = len(fires)
        avg_frp = fires["frp"].mean()
        days = fires["acq_date"].nunique()

        high_frp = len(
            fires[fires["frp"] >= fires["frp"].quantile(0.90)]
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            show_kpi(
                "SATELLITE DETECTIONS",
                f"{total:,}",
                "NASA VIIRS observations"
            )

        with c2:
            show_kpi(
                "AVERAGE FIRE POWER",
                f"{avg_frp:.2f} MW",
                "Mean FRP"
            )

        with c3:
            show_kpi(
                "OBSERVATION DAYS",
                f"{days:,}",
                "Unique observation dates"
            )

        with c4:
            show_kpi(
                "TOP 10% FRP RECORDS",
                f"{high_frp:,}",
                "Highest FRP observations"
            )

    else:
        st.warning("Fire dataset unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Intelligence Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="panel">
            <div style="font-size:29px;">🛰️</div>
            <div class="panel-title">Satellite Monitoring</div>
            <p class="panel-description">
                Analyse real NASA satellite fire observations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
            <div style="font-size:29px;">📍</div>
            <div class="panel-title">Hotspot Intelligence</div>
            <p class="panel-description">
                Identify geographical areas with repeated detections.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="panel">
            <div style="font-size:29px;">🤖</div>
            <div class="panel-title">Predictive Analytics</div>
            <p class="panel-description">
                Estimate next-day satellite detection likelihood.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Historical Activity")

    if monthly is not None:

        fig = px.bar(
            monthly,
            x="month",
            y="fire_detections",
            color_discrete_sequence=["#23865A"],
            title="Monthly Fire Detection Distribution"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.info(
        "NASA satellite detections are not automatically confirmed "
        "agricultural residue-burning incidents."
    )
    # =====================================================
# PAGE 2: GEOSPATIAL INTELLIGENCE
# =====================================================

elif page == "Geospatial Intelligence":

    show_hero(
        "Geospatial Intelligence",
        "Explore geographical concentrations of historical satellite fire detections.",
        "LOCATION ANALYTICS"
    )

    map_path = MAP_DIR / "punjab_haryana_fire_map.html"

    if map_path.exists():

        map_html = map_path.read_text(encoding="utf-8")

        st.components.v1.html(
            map_html,
            height=750,
            scrolling=True
        )

    else:
        st.warning(
            "Map not found. Run src/create_interactive_map.py."
        )

    st.caption(
        "Map represents historical satellite detections. "
        "It does not independently confirm stubble burning."
    )


# =====================================================
# PAGE 3: HISTORICAL ANALYTICS
# =====================================================

elif page == "Historical Analytics":

    show_hero(
        "Historical Analytics",
        "Discover seasonal patterns, annual variation and fire intensity.",
        "DATA EXPLORATION"
    )

    if monthly is not None:

        st.markdown("### Monthly Detection Distribution")

        fig = px.bar(
            monthly,
            x="month",
            y="fire_detections",
            color="fire_detections",
            color_continuous_scale=[
                "#D9F0DF",
                "#6DBF91",
                "#176B45"
            ],
            title="Monthly Satellite Fire Detections"
        )

        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    if yearly is not None:

        st.markdown("### Annual Detection Distribution")

        fig = px.line(
            yearly,
            x="year",
            y="fire_detections",
            markers=True,
            title="Year-wise Satellite Fire Detections"
        )

        fig.update_traces(
            line_color="#176B45",
            line_width=3,
            marker_size=9
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    if fires is not None:

        st.markdown("### Fire Radiative Power Distribution")

        fig = px.histogram(
            fires,
            x="frp",
            nbins=50,
            color_discrete_sequence=["#23865A"],
            title="Fire Radiative Power (FRP)"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    if hotspots is not None:

        st.markdown("### Historical Hotspot Records")

        st.dataframe(
            hotspots.head(20),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# PAGE 4: AI MODEL INSIGHTS
# =====================================================

elif page == "AI Model Insights":

    show_hero(
        "AI Model Insights",
        "Evaluate the performance of our machine learning prediction model.",
        "MACHINE LEARNING"
    )

    if metrics is not None:

        st.markdown("### Model Evaluation")

        c1, c2, c3 = st.columns(3)

        with c1:
            show_kpi(
                "ACCURACY",
                f"{metrics['accuracy'] * 100:.2f}%",
                "Overall correct predictions"
            )

        with c2:
            show_kpi(
                "PRECISION",
                f"{metrics['precision'] * 100:.2f}%",
                "Positive prediction reliability"
            )

        with c3:
            show_kpi(
                "RECALL",
                f"{metrics['recall'] * 100:.2f}%",
                "Positive cases identified"
            )

        c4, c5 = st.columns(2)

        with c4:
            show_kpi(
                "F1 SCORE",
                f"{metrics['f1_score'] * 100:.2f}%",
                "Precision-recall balance"
            )

        with c5:

            auc = metrics.get("roc_auc")

            show_kpi(
                "ROC-AUC",
                f"{auc * 100:.2f}%" if auc is not None else "N/A",
                "Class discrimination"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Evaluation Period")

        a, b = st.columns(2)

        a.info(f"Training years: {metrics['training_years']}")
        b.info(f"Testing year: {metrics['testing_year']}")

        if importance is not None:

            st.markdown("### Feature Importance")

            fig = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[
                    "#B5D99C",
                    "#176B45"
                ],
                title="Model Feature Importance"
            )

            fig.update_layout(coloraxis_showscale=False)

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        st.warning(
            "The current model predicts next-day satellite fire detection. "
            "It is not yet a validated crop-residue-burning predictor."
        )

    else:
        st.error("Model evaluation metrics unavailable.")


# =====================================================
# PAGE 5: PREDICTION CENTRE
# =====================================================

elif page == "Prediction Centre":

    show_hero(
        "Next-Day Prediction Centre",
        "Estimate the likelihood of a satellite fire detection "
        "in a selected geographical grid cell.",
        "PREDICTIVE INTELLIGENCE"
    )

    if model is None:

        st.error(
            "Model not found. Run src/train_model.py first."
        )

    else:

        st.markdown("### Location Parameters")

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
                "Observation date",
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
            "GENERATE AI PREDICTION",
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

            st.success("Prediction generated.")

            st.markdown("### Prediction Result")

            st.metric(
                "Next-Day Detection Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            st.caption(
                "Model-estimated probability of a satellite detection. "
                "This is not a calibrated probability of stubble burning."
            )



# =====================================================
# PAGE 6: AI FARMING ASSISTANT
# =====================================================

elif page == "AI Farming Assistant":

    show_hero(
        "AI Farming Assistant",
        "Get practical agricultural guidance powered by Gemini AI.",
        "PARALILOOP AI AGENT"
    )

    st.markdown("### 🌾 Ask ParaliLoop AI")

    st.caption(
        "Ask questions about stubble management, crop residue, "
        "Happy Seeder, Super Seeder and sustainable farming."
    )

    if "paraliloop_chat_history" not in st.session_state:
        st.session_state.paraliloop_chat_history = []

    st.markdown("#### Suggested questions")

    suggestions = [
        "What are alternatives to stubble burning?",
        "How does a Happy Seeder work?",
        "How can farmers manage crop residue?"
    ]

    cols = st.columns(3)

    for i, question in enumerate(suggestions):
        with cols[i]:
            if st.button(
                question,
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                st.session_state.pending_farmer_question = question

    # Display conversation history
    for message in st.session_state.paraliloop_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask your farming question in Hindi or English..."
    )

    if "pending_farmer_question" in st.session_state:
        prompt = st.session_state.pop("pending_farmer_question")

    if prompt:

        with st.chat_message("user"):
            st.markdown(prompt)

        farm_context = (
            "ParaliLoop AI analyses historical satellite fire "
            "detections across Punjab and Haryana. These detections "
            "do not independently confirm stubble burning."
        )

        if fires is not None:
            farm_context += (
                f"\nHistorical dataset records: {len(fires):,}"
            )

        with st.chat_message("assistant"):

            with st.spinner("ParaliLoop AI is thinking..."):

                answer = ask_gemini(
                    prompt,
                    chat_history=st.session_state.paraliloop_chat_history,
                    farm_context=farm_context
                )

            st.markdown(answer)

        st.session_state.paraliloop_chat_history.append({
            "role": "user",
            "content": prompt
        })

        st.session_state.paraliloop_chat_history.append({
            "role": "assistant",
            "content": answer
        })

    if st.session_state.paraliloop_chat_history:

        if st.button("Clear Chat", key="clear_paraliloop_chat"):
            st.session_state.paraliloop_chat_history = []
            st.rerun()


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    <b>PARALILOOP AI</b><br>
    Team TriVortex · Sustainability Hackathon · IIT Indore<br>
    Satellite Intelligence | Machine Learning | Agricultural Sustainability
</div>
""", unsafe_allow_html=True)