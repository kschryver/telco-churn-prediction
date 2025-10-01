# Telco Customer Churn Prediction

Predict which customers will churn and recommend specific retention actions based on their churn drivers.

## Overview

Built an XGBoost model on Kaggle's Telco dataset (7k customers, 26.5% churn rate). Uses SHAP to map predictions to interventions:

- Price-sensitive customers → discount offers
- Low engagement → onboarding campaigns  
- Service issues → priority support
- High-value at risk → account manager

## Tech Stack

Python • pandas • scikit-learn • XGBoost • SHAP • Streamlit

## Deliverables

- Churn prediction model (targeting 70%+ precision, high recall)
- Intervention recommendation engine
- Interactive Streamlit dashboard with risk scoring and what-if simulator
- Business impact analysis (CLV, ROI by intervention type)

## Key Findings

- Month-to-month contracts: 42% churn rate
- Two-year contracts: 3% churn rate
- Churners have 60% lower tenure and higher monthly charges
- Senior citizens show elevated churn risk

## Status

- [x] EDA + data cleaning
- [ ] Feature engineering + baseline model
- [ ] Model optimization
- [ ] Intervention logic (SHAP)
- [ ] Business metrics + ROI
- [ ] Dashboard build + deployment