# Telco Churn Prediction + Intervention System

Churn model with SHAP-based intervention recommendations. 72% recall, $371K retained revenue, 1,395% ROI.

## Dataset

Kaggle Telco Customer Churn - 7,043 customers, 26.5% churn rate

## Results

**Model Performance**
- XGBoost: 72% recall, 54% precision, 0.836 ROC-AUC
- Baseline (Logistic Regression): 56% recall, 66% precision, 0.842 ROC-AUC
- Catches 270/374 churners (vs 209/374 baseline)

**Business Impact**
- 497 at-risk customers identified
- 199 expected retentions (40% success rate)
- $26,600 intervention cost
- $397,600 revenue retained
- **$371,000 net benefit (1,395% ROI)**

**Intervention Breakdown**
- Contract upgrades: 477 customers (96%)
- Fiber bundle discounts: 18 customers
- Auto-pay migration: 2 customers

## Approach

**Modeling**: XGBoost optimized for recall using `scale_pos_weight=2.5` to catch more churners

**SHAP Analysis**: Maps predictions to root causes (contract type, tenure, pricing) for each customer

**Intervention Logic**:
- Month-to-month contract → upgrade offer
- High monthly charges + fiber optic → bundle discount
- Electronic check payment → auto-pay migration
- Low service adoption → onboarding campaign

## Key Findings

**Top Churn Drivers (SHAP)**:
1. Tenure (new customers highest risk)
2. Contract type (month-to-month = 42% churn)
3. Monthly charges (price sensitivity)
4. Fiber optic internet (service quality vs price perception)
5. Electronic check payment (friction)

**Highest Risk Segment**: Customers with tenure < 12 months + month-to-month contracts

## Tech Stack

Python • pandas • scikit-learn • XGBoost • SHAP • matplotlib • seaborn

## Status

**Completed**
- EDA with segment analysis
- Feature engineering (tenure buckets, service counts, price ratios)
- Baseline logistic regression (81% accuracy, 0.84 AUC)
- XGBoost optimization (72% recall)
- SHAP-based intervention mapping
- Business impact calculations

**Next**
- Streamlit dashboard with risk scores and action recommendations
- What-if simulator for testing strategies
- Deployment to Streamlit Cloud

## Files

```
├── Telco-Customer-Churn.ipynb              # EDA + modeling
├── Telco-Customer-Churn-Clean.csv          # Cleaned dataset
├── xgboost_model.pkl                       # Trained XGBoost model
├── shap_values.pkl                         # SHAP explanations
├── at_risk_customers_with_interventions.csv # Action recommendations
└── README.md
```