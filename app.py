import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier

# Load saved model and data
@st.cache_resource
def load_model():
    with open('xgboost_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('at_risk_customers_with_interventions.csv')

model = load_model()
at_risk_df = load_data()

# Page config
st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")
st.title("📊 Customer Churn Risk Dashboard")

# Sidebar navigation
page = st.sidebar.radio("Navigate", 
                        ["Overview", "Risk Scoring", "Interventions", "Business Impact"])

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total At-Risk Customers", f"{len(at_risk_df):,}")
    with col2:
        st.metric("Avg Risk Score", f"{at_risk_df['risk_score'].mean():.1%}")
    with col3:
        st.metric("Expected Retentions (40%)", f"{int(len(at_risk_df) * 0.4)}")
    with col4:
        st.metric("Expected Revenue Retained", f"${int(len(at_risk_df) * 0.4 * 2000):,}")
    
    st.divider()
    
    # Risk distribution
    fig = px.histogram(at_risk_df, x='risk_score', nbins=30, 
                       title="Distribution of Customer Risk Scores",
                       labels={'risk_score': 'Risk Score', 'count': 'Number of Customers'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Intervention breakdown
    intervention_counts = at_risk_df['intervention'].value_counts()
    fig = px.bar(intervention_counts, 
                 title="Customers by Recommended Intervention",
                 labels={'value': 'Count', 'index': 'Intervention Type'})
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: RISK SCORING
# ============================================================================
elif page == "Risk Scoring":
    st.header("Individual Customer Risk Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        min_risk = st.slider("Filter by minimum risk score:", 0.0, 1.0, 0.5)
    with col2:
        intervention_filter = st.multiselect("Filter by intervention type:", 
                                            at_risk_df['intervention'].unique(),
                                            default=at_risk_df['intervention'].unique())
    
    filtered_df = at_risk_df[(at_risk_df['risk_score'] >= min_risk) & 
                             (at_risk_df['intervention'].isin(intervention_filter))]
    
    st.subheader(f"Showing {len(filtered_df)} customers")
    
    # Display top 20 customers
    display_cols = ['risk_score', 'tenure', 'MonthlyCharges', 'num_services', 'intervention']
    st.dataframe(filtered_df[display_cols].head(20).sort_values('risk_score', ascending=False),
                 use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "at_risk_customers.csv", "text/csv")

# ============================================================================
# PAGE 3: INTERVENTIONS
# ============================================================================
elif page == "Interventions":
    st.header("Intervention Strategy")
    
    intervention_data = []
    intervention_costs = {
        'Contract Upgrade': 50,
        'Fiber Bundle Discount': 150,
        'Auto-Pay Migration': 25,
        'Onboarding Campaign': 30,
        'Priority Support': 100
    }
    
    for intervention in at_risk_df['intervention'].unique():
        count = len(at_risk_df[at_risk_df['intervention'] == intervention])
        cost = intervention_costs.get(intervention, 75)
        avg_risk = at_risk_df[at_risk_df['intervention'] == intervention]['risk_score'].mean()
        
        intervention_data.append({
            'Intervention': intervention,
            'Customer Count': count,
            'Cost per Customer': cost,
            'Total Cost': count * cost,
            'Avg Risk Score': f"{avg_risk:.1%}"
        })
    
    intervention_df = pd.DataFrame(intervention_data).sort_values('Customer Count', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(intervention_df, x='Intervention', y='Customer Count',
                    title="Customers per Intervention",
                    labels={'Customer Count': 'Number of Customers'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(intervention_df, x='Intervention', y='Total Cost',
                    title="Total Cost by Intervention",
                    labels={'Total Cost': 'Cost ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(intervention_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 4: BUSINESS IMPACT
# ============================================================================
elif page == "Business Impact":
    st.header("ROI & Business Impact")
    
    avg_clv = 2000
    retention_rate = st.slider("Assumed retention success rate:", 0.0, 1.0, 0.4, 0.05)
    
    intervention_costs = {
        'Contract Upgrade': 50,
        'Fiber Bundle Discount': 150,
        'Auto-Pay Migration': 25,
        'Onboarding Campaign': 30,
        'Priority Support': 100
    }
    
    # Calculate by intervention
    impact_data = []
    for intervention in at_risk_df['intervention'].unique():
        count = len(at_risk_df[at_risk_df['intervention'] == intervention])
        cost_per = intervention_costs.get(intervention, 75)
        total_cost = count * cost_per
        customers_saved = count * retention_rate
        revenue_retained = customers_saved * avg_clv
        roi = (revenue_retained - total_cost) / total_cost if total_cost > 0 else 0
        
        impact_data.append({
            'Intervention': intervention,
            'Customers': count,
            'Total Cost': total_cost,
            'Customers Saved': int(customers_saved),
            'Revenue Retained': int(revenue_retained),
            'ROI': f"{roi:.0%}"
        })
    
    impact_df = pd.DataFrame(impact_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Program Cost", f"${impact_df['Total Cost'].sum():,.0f}")
    with col2:
        st.metric("Expected Customers Saved", f"{impact_df['Customers Saved'].sum():,}")
    with col3:
        st.metric("Revenue Retained", f"${impact_df['Revenue Retained'].sum():,.0f}")
    with col4:
        total_saved = impact_df['Revenue Retained'].sum()
        total_cost = impact_df['Total Cost'].sum()
        overall_roi = (total_saved - total_cost) / total_cost if total_cost > 0 else 0
        st.metric("Overall ROI", f"{overall_roi:.0%}")
    
    st.divider()
    
    # Impact visualization
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(impact_df, x='Intervention', y='Revenue Retained',
                    title="Revenue Retained by Intervention")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(impact_df, x='Intervention', y='Customers Saved',
                    title="Customers Saved by Intervention")
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    st.info(f"💡 At {retention_rate:.0%} success rate, the program generates ${total_saved - total_cost:,.0f} in net value")