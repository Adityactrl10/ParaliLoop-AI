# 🌾 ParaliLoop AI
### Intelligent Satellite-Based Crop Residue Fire Detection & Agricultural Assistant

**Developed by Team TriVortex | Sustainability Hackathon | IIT Indore**

## 📌 Overview

ParaliLoop AI is an AI-powered agricultural sustainability platform designed to analyze satellite-based fire detections across Punjab and Haryana.

The platform combines machine learning, geospatial intelligence, historical fire analysis and a Gemini-powered AI Farming Assistant to support better crop residue management decisions.

## 🎯 Problem Statement

Crop residue burning is a major environmental concern in northern India. It contributes to air pollution, greenhouse gas emissions and soil degradation.

Farmers also face challenges in accessing practical information about sustainable crop residue management alternatives.

ParaliLoop AI aims to address these challenges through satellite data analysis, fire-risk prediction and AI-powered agricultural guidance.

## ✨ Key Features

- **Satellite Fire Intelligence:** Visualizes satellite-based fire detections across Punjab and Haryana.
- **Interactive Geospatial Map:** Displays district-level fire activity and geographical patterns.
- **Historical Analytics:** Analyzes yearly and monthly fire trends.
- **Machine Learning Prediction:** Estimates the likelihood of future satellite fire detections using historical patterns.
- **AI Farming Assistant:** Gemini-powered assistant providing agricultural guidance in Hindi and English, with Punjabi support.
- **Crop Residue Management:** Provides information about Happy Seeder, Super Seeder, mulching, biomass utilization and other alternatives.

## 🧠 Machine Learning

The project uses historical satellite fire data to train a machine learning model.

**Dataset split:**
- Training period: 2021–2024
- Testing period: 2025

**Model evaluation:**

| Metric | Score |
|---|---:|
| Accuracy | 82.57% |
| Precision | 32.27% |
| Recall | 54.48% |
| F1 Score | 40.53% |
| ROC AUC | 79.27% |

*Note: Model predictions indicate satellite fire-detection likelihood and do not independently confirm stubble burning.*

## 🛠️ Technology Stack

- **Programming:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Dashboard:** Streamlit
- **Geospatial Visualization:** Folium, GeoPandas
- **AI Integration:** Google Gemini API
- **Data Visualization:** Matplotlib
- **Version Control:** Git and GitHub
- **Large Model Storage:** Git LFS

## 📂 Project Structure

```text
ParaliLoop-AI/
│
├── dashboard/
│   ├── maps/
│   ├── visualizations/
│   ├── app.py
│   └── pro_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── boundaries/
│
├── models/
│   ├── paraliloop_model.pkl
│   ├── model_metrics.json
│   └── feature_importance.csv
│
├── src/
│   ├── ai_agent.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── risk_analysis.py
│   └── other processing scripts
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Adityactrl10/ParaliLoop-AI.git
cd ParaliLoop-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Never upload your API key or `.env` file to GitHub.

### 6. Run the dashboard

```bash
streamlit run dashboard/pro_app.py --server.port 8502
```

Open `http://localhost:8502` in your browser.

## 🌱 Expected Impact

- Improved understanding of satellite fire-detection patterns.
- Better awareness of sustainable crop residue management.
- Data-driven agricultural insights.
- Accessible AI-powered farming guidance.
- Support for environmentally sustainable agricultural practices.

## ⚠️ Limitations

- Satellite detections are not independently verified incidents of stubble burning.
- Model predictions are estimates and may contain false positives and false negatives.
- AI-generated agricultural guidance should be verified with qualified local agricultural experts.
- The platform does not claim access to live satellite or weather data.

## 👥 Team

**Team TriVortex**

Developed for the Sustainability Hackathon at IIT Indore.

## 📜 Disclaimer

ParaliLoop AI is an academic and sustainability-focused project. It is not a government authority or a substitute for professional agricultural advice.

---

**ParaliLoop AI – Turning Satellite Intelligence into Sustainable Agricultural Action.** 🌍
