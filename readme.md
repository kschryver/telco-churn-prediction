# Telco Churn Prediction + Intervention System

Churn model that recommends retention actions based on risk drivers.

## Dataset

Kaggle Telco Customer Churn - 7,043 customers, 21 features, 26.5% churn rate.

## Approach

**Modeling**: Baseline logistic regression (81% accuracy, 0.84 AUC, 56% recall). XGBoost planned for improved recall.

**Intervention Logic**: Map SHAP values to actions:
- High monthly charges → price discount
- Low service adoption → onboarding campaign  
- Service quality issues → priority support
- High CLV at risk → account manager

**Business Metrics**: CLV calculations, ROI per intervention type, segment analysis.

**Dashboard**: Streamlit app with risk scores, action recommendations, what-if simulator.

## Tech Stack

Python • pandas • scikit-learn • XGBoost • SHAP • Streamlit

## Key Findings

- Month-to-month contracts: 42% churn vs 3% for two-year
- Fiber optic customers: 42% churn (price sensitivity)
- Electronic check payment: 45% churn
- No tech support: 42% churn
- Top predictors: contract type, tenure, monthly charges

## Status

**Completed**
- EDA with segment analysis
- Feature engineering (tenure buckets, service counts, price ratios)
- Baseline model (logistic regression)

**Next**
- XGBoost optimization
- SHAP-based intervention mapping
- Business impact calculations
- Streamlit dashboard + deployment

## Structure

```
├── Telco-Customer-Churn.ipynb          # EDA + baseline model
├── Telco-Customer-Churn-Clean.csv      # Cleaned dataset
├── baseline_model.pkl                  # Saved logistic regression
├── scaler.pkl                          # Feature scaler
└── README.md
```