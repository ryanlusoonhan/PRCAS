"""
Streamlit Dashboard - Churn-Shield AI UI

This module implements a Streamlit UI that:
- Connects to the FastAPI backend
- Displays a "What-If" simulator (sliders for usage metrics)
- Renders SHAP force plots for local interpretability
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import yaml

# Page configuration
st.set_page_config(
    page_title="Churn-Shield AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return {}

config = load_config()
dashboard_config = config.get('dashboard', {})
api_url = dashboard_config.get('api_url', 'http://localhost:8000')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9900;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00aa00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .feature-impact {
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"customer_data": data},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def get_explanation(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get SHAP explanation from API."""
    try:
        response = requests.post(
            f"{api_url}/explain",
            json={"customer_data": data},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def render_gauge(probability: float, risk_level: str):
    """Render a gauge chart for churn probability."""
    color = "#00aa00" if risk_level == "Low" else "#ff9900" if risk_level == "Medium" else "#ff0000"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        number={'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ccffcc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ffcccc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig


def render_feature_importance(top_features: list):
    """Render feature importance bar chart."""
    features = [f['feature'] for f in top_features]
    impacts = [f['impact'] for f in top_features]
    colors = ['#ff6b6b' if f['direction'] == 'increases' else '#4ecdc4' for f in top_features]
    
    fig = go.Figure(data=[
        go.Bar(
            x=impacts,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{imp:.3f}" for imp in impacts],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top Risk Drivers",
        xaxis_title="Impact (SHAP Value)",
        yaxis_title="Feature",
        height=300,
        showlegend=False
    )
    
    return fig


def render_shap_waterfall(explanation: Dict[str, Any]):
    """Render SHAP waterfall plot."""
    if not explanation:
        return None
    
    # Get top 10 features by absolute SHAP value
    shap_values = np.array(explanation['shap_values'])
    feature_names = explanation['feature_names']
    feature_values = explanation['feature_values']
    base_value = explanation['base_value']
    
    # Get indices of top 10 features by absolute SHAP value
    top_indices = np.argsort(np.abs(shap_values))[-10:]
    
    # Prepare data for waterfall
    measure = ["relative"] * len(top_indices) + ["total"]
    x = [shap_values[i] for i in top_indices] + [base_value + sum(shap_values[top_indices])]
    text = [f"{feature_names[i]}<br>{feature_values[i]:.2f}" for i in top_indices] + ["Final Prediction"]
    
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measure,
        x=text,
        y=x,
        textposition="outside",
        text=[f"{v:+.3f}" for v in x],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#ff6b6b"}},
        decreasing={"marker": {"color": "#4ecdc4"}},
        totals={"marker": {"color": "#45b7d1"}}
    ))
    
    fig.update_layout(
        title="SHAP Explanation (Feature Contributions)",
        showlegend=False,
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Churn-Shield AI Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_available = check_api_health()
    
    if not api_available:
        st.error("⚠️ API is not available. Please ensure the FastAPI server is running.")
        st.info(f"Expected API at: {api_url}")
        return
    
    st.success("✅ API Connection Established")
    
    # Sidebar - Customer Input
    st.sidebar.header("📊 Customer Data Input")
    st.sidebar.markdown("Enter customer details or use the What-If simulator below.")
    
    # State mapping from config
    state_to_region = config.get('state_to_region', {})
    states = sorted(list(set(state_to_region.keys())))
    
    # Basic Information
    st.sidebar.subheader("Basic Information")
    state = st.sidebar.selectbox("State", states, index=states.index('CA') if 'CA' in states else 0)
    account_length = st.sidebar.slider("Account Length (months)", 1, 250, 100)
    area_code = st.sidebar.selectbox("Area Code", [408, 415, 510], index=1)
    
    # Plan Information
    st.sidebar.subheader("Plan Information")
    international_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"], index=0)
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"], index=1)
    number_vmail_messages = st.sidebar.slider("Voicemail Messages", 0, 50, 25)
    
    # Usage Metrics - What-If Simulator
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎮 What-If Simulator")
    st.sidebar.markdown("Adjust usage metrics to see impact on churn risk.")
    
    # Day usage
    st.sidebar.markdown("**Day Usage**")
    total_day_minutes = st.sidebar.slider("Day Minutes", 0.0, 500.0, 265.0, step=0.1)
    total_day_calls = st.sidebar.slider("Day Calls", 0, 200, 110)
    total_day_charge = total_day_minutes * 0.17  # Approximate rate
    
    # Evening usage
    st.sidebar.markdown("**Evening Usage**")
    total_eve_minutes = st.sidebar.slider("Evening Minutes", 0.0, 400.0, 200.0, step=0.1)
    total_eve_calls = st.sidebar.slider("Evening Calls", 0, 200, 100)
    total_eve_charge = total_eve_minutes * 0.085  # Approximate rate
    
    # Night usage
    st.sidebar.markdown("**Night Usage**")
    total_night_minutes = st.sidebar.slider("Night Minutes", 0.0, 400.0, 200.0, step=0.1)
    total_night_calls = st.sidebar.slider("Night Calls", 0, 200, 100)
    total_night_charge = total_night_minutes * 0.045  # Approximate rate
    
    # International usage
    st.sidebar.markdown("**International Usage**")
    total_intl_minutes = st.sidebar.slider("International Minutes", 0.0, 30.0, 10.0, step=0.1)
    total_intl_calls = st.sidebar.slider("International Calls", 0, 20, 4)
    total_intl_charge = total_intl_minutes * 0.27  # Approximate rate
    
    # Customer service
    st.sidebar.markdown("**Customer Service**")
    customer_service_calls = st.sidebar.slider("Service Calls", 0, 20, 1)
    
    # Prepare customer data
    customer_data = {
        "state": state,
        "account_length": account_length,
        "area_code": area_code,
        "international_plan": international_plan,
        "voice_mail_plan": voice_mail_plan,
        "number_vmail_messages": number_vmail_messages,
        "total_day_minutes": round(total_day_minutes, 2),
        "total_day_calls": total_day_calls,
        "total_day_charge": round(total_day_charge, 2),
        "total_eve_minutes": round(total_eve_minutes, 2),
        "total_eve_calls": total_eve_calls,
        "total_eve_charge": round(total_eve_charge, 2),
        "total_night_minutes": round(total_night_minutes, 2),
        "total_night_calls": total_night_calls,
        "total_night_charge": round(total_night_charge, 2),
        "total_intl_minutes": round(total_intl_minutes, 2),
        "total_intl_calls": total_intl_calls,
        "total_intl_charge": round(total_intl_charge, 2),
        "customer_service_calls": customer_service_calls
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Prediction Results")
        
        # Get prediction
        with st.spinner("Analyzing customer risk..."):
            prediction = get_prediction(customer_data)
        
        if prediction:
            # Display risk level with appropriate styling
            risk_class = f"risk-{prediction['risk_level'].lower()}"
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>Risk Level: {prediction['risk_level']}</h3>
                <p><strong>Recommendation:</strong> {prediction['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Churn Probability", f"{prediction['churn_probability']*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prediction", "Will Churn" if prediction['prediction'] == 1 else "Will Stay")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Model Version", prediction.get('model_version', '1.0.0'))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Gauge chart
            st.plotly_chart(
                render_gauge(prediction['churn_probability'], prediction['risk_level']),
                use_container_width=True
            )
            
            # Top risk drivers
            st.subheader("🔍 Top Risk Drivers")
            st.plotly_chart(
                render_feature_importance(prediction['top_features']),
                use_container_width=True
            )
            
            # Explanation
            with st.expander("📊 Detailed SHAP Explanation"):
                with st.spinner("Generating explanation..."):
                    explanation = get_explanation(customer_data)
                
                if explanation:
                    st.plotly_chart(
                        render_shap_waterfall(explanation),
                        use_container_width=True
                    )
                    
                    # Show raw values
                    st.markdown("**Feature Values:**")
                    feature_df = pd.DataFrame({
                        'Feature': explanation['feature_names'][:10],
                        'Value': [f"{v:.2f}" for v in explanation['feature_values'][:10]],
                        'SHAP Value': [f"{v:+.4f}" for v in explanation['shap_values'][:10]]
                    })
                    st.dataframe(feature_df, use_container_width=True)
    
    with col2:
        st.subheader("📋 Customer Profile")
        
        # Display customer summary
        st.markdown(f"""
        **Location:** {state} ({state_to_region.get(state, 'Unknown')})
        
        **Account:** {account_length} months
        
        **Plans:**
        - International: {international_plan}
        - Voicemail: {voice_mail_plan}
        
        **Usage Summary:**
        - Day: {total_day_minutes:.1f} min ({total_day_calls} calls)
        - Evening: {total_eve_minutes:.1f} min ({total_eve_calls} calls)
        - Night: {total_night_minutes:.1f} min ({total_night_calls} calls)
        - International: {total_intl_minutes:.1f} min ({total_intl_calls} calls)
        
        **Service Calls:** {customer_service_calls}
        """)
        
        # Churn reduction potential
        st.markdown("---")
        st.subheader("💡 Churn Reduction Tips")
        
        tips = []
        if customer_service_calls >= 3:
            tips.append("⚠️ High service call volume - consider proactive customer service outreach")
        if international_plan == "Yes" and total_intl_charge > 5:
            tips.append("💰 High international charges - consider offering international plan discount")
        if total_day_charge > 45:
            tips.append("📞 High day usage charges - consider unlimited day plan offer")
        if number_vmail_messages == 0 and voice_mail_plan == "Yes":
            tips.append("📧 Paying for voicemail but not using it - consider plan downgrade")
        
        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success("✅ No immediate risk factors identified")
        
        # What-If Analysis
        st.markdown("---")
        st.subheader("🎯 What-If Analysis")
        
        st.markdown("""
        Adjust the sliders in the sidebar to simulate different scenarios:
        
        1. **Reduce Service Calls:** Try setting service calls to 0-1
        2. **Add International Plan:** See how removing high intl charges affects risk
        3. **Change Usage Patterns:** Adjust day/evening/night minutes
        
        The prediction will update automatically!
        """)


if __name__ == "__main__":
    main()
