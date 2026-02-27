# ⚽ World Cup Match Predictor

An ML-powered web application that predicts international football match outcomes using historical data spanning 150+ years (1872-2026).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

## Overview

This project analyses 49,071 international football matches to predict match outcomes between any two national teams. It features an interactive Streamlit dashboard with win probabilities, team comparisons, head-to-head records, and radar charts.

## Features

- **Match Prediction**: Select any two teams and get win/draw/loss probabilities
- **Team Form Analysis**: Last 30 matches breakdown with win rates, goals scored/conceded
- **Head-to-Head Records**: Historical matchup data between selected teams
- **Radar Comparison**: Visual comparison across multiple performance metrics
- **Tournament Weighting**: World Cup matches weighted higher than friendlies

## How It Works

1. **Feature Engineering**: For each match, the model calculates team form stats (win rate, goals, weighted points) from their last 30 matches, plus head-to-head records
2. **Tournament Weighting**: Competitive matches (World Cup = 1.0) are valued higher than friendlies (0.4)
3. **XGBoost Classifier**: Trained on 19,780 matches from 2000-2026 to predict Home Win, Draw, or Away Win
4. **Model Accuracy**: 58% on three-way classification (Home Win / Draw / Away Win), which is competitive with professional betting models (52-60% typical range)

## Project Structure
```
world-cup-predictor/
├── data/
│   └── results.csv              # 49,071 international matches (1872-2026)
├── notebooks/
│   └── fifa_predictor.ipynb     # Full analysis notebook
├── app/
│   └── dashboard.py             # Streamlit dashboard
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   └── label_encoder.pkl       # Label encoder for predictions
├── requirements.txt
└── README.md
```

## Quick Start
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/world-cup-predictor.git
cd world-cup-predictor

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/dashboard.py
```

## Dataset

- **Source**: Kaggle - International Football Results
- **Matches**: 49,071
- **Date Range**: 1872-11-30 to 2026-01-26
- **Teams**: 325 national teams
- **Features**: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

## Tech Stack

- **Python** - Core language
- **Pandas & NumPy** - Data manipulation
- **XGBoost** - Machine learning model
- **Scikit-learn** - Model evaluation and preprocessing
- **Streamlit** - Interactive dashboard
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Exploratory analysis charts

## Key Insights from EDA

- Home teams win approximately 46% of matches, confirming home advantage
- Most matches produce 2-3 total goals
- Friendlies make up 37% of all matches — weighting them lower improved model performance
- Draws are the hardest outcome to predict (only 6% recall) — this is a known challenge in football prediction

## Future Improvements

- Add Elo rating system as a feature
- Implement time-decay weighting (recent matches matter more)
- Add player-level data for richer predictions
- Deploy dashboard to Streamlit Cloud for public access
- Hyperparameter tuning with cross-validation

## Author

Built as part of a BSc Honours in Data Science portfolio project.

## License

MIT License
