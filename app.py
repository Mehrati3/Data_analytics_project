import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Page Configuration (Modern & Minimalist) ---
st.set_page_config(page_title="ICU Risk Analytics", layout="wide")

# Custom CSS for the refined, professional aesthetic
st.markdown("""
    <style>
    .main { background-color: #f5f7f5; }
    .stButton>button { 
        background-color: #556b2f; 
        color: white; 
        border-radius: 5px; 
        width: 100%;
        font-weight: bold;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    .calc-box {
        border: 2px solid #556b2f; 
        padding: 20px; 
        border-radius: 10px; 
        background-color: #ffffff;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Multi-Model Logic Functions (Requirement 2 & 3) ---

def calculate_weighted_risk(hr, temp, spo2, wbc, creatinine):
    """Model 1: Transparent Weighted Point System [cite: 9, 21]"""
    points = {} 
    # Heart Rate Points
    if hr > 120 or hr < 50: points['Heart Rate'] = 2
    elif hr > 100 or hr < 60: points['Heart Rate'] = 1
    else: points['Heart Rate'] = 0
    # Temperature Points
    if temp > 38.5 or temp < 36.0: points['Temperature'] = 2
    elif temp > 37.5 or temp < 36.5: points['Temperature'] = 1
    else: points['Temperature'] = 0
    # SpO2 Points
    if spo2 < 90: points['Oxygen (SpO2)'] = 3
    elif spo2 < 94: points['Oxygen (SpO2)'] = 1
    else: points['Oxygen (SpO2)'] = 0
    # WBC Count Points
    if wbc > 15.0 or wbc < 4.0: points['WBC Count'] = 2
    elif wbc > 11.0: points['WBC Count'] = 1
    else: points['WBC Count'] = 0
    # Creatinine Points
    if creatinine > 1.5: points['Creatinine'] = 1
    else: points['Creatinine'] = 0

    score_sum = sum(points.values())
    risk_percentage = (score_sum / 10) * 100
    
    if risk_percentage >= 70:
        status, color, advice = "CRITICAL", "red", "High risk detected. Immediate clinical intervention required."
    elif risk_percentage >= 30:
        status, color, advice = "MONITOR", "orange", "Early warning signs. Increase monitoring frequency."
    else:
        status, color, advice = "STABLE", "green", "Patient metrics within acceptable clinical ranges."
        
    return risk_percentage, status, color, advice, points

def threshold_binary_model(hr, temp, spo2, wbc, creatinine):
    """Model 2: Strict Safety Thresholds [cite: 20]"""
    critical = 1 if (hr > 130 or spo2 < 88 or temp > 39.0 or wbc > 18.0) else 0
    risk = 85.0 if critical > 0 else 15.0
    return risk, "Threshold Binary"

def linear_coefficient_model(hr, temp, spo2, wbc, creatinine):
    """Model 3: Coefficient-Based Weighting [cite: 9]"""
    risk = (0.2 * hr) + (0.3 * (100 - spo2)) + (0.1 * wbc) + (0.4 * creatinine)
    risk = min(risk, 100)
    return risk, "Coefficient Sum"

# --- 3. Sidebar: Data Collection (Requirement 1) ---
st.sidebar.header("Clinical Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Patient CSV", type=["csv"])

st.sidebar.markdown("---")
with st.sidebar.form("input_form"):
    st.write("**Manual Entry**")
    m_hr = st.number_input("Heart Rate (bpm)", 30, 250, 80)
    m_temp = st.number_input("Body Temp (°C)", 30.0, 45.0, 37.0)
    m_spo2 = st.slider("SpO2 (%)", 50, 100, 95)
    m_wbc = st.number_input("WBC Count (10^9/L)", 0.0, 50.0, 10.0)
    m_creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0)
    submit = st.form_submit_button("Run Analysis & Compare")

# --- 4. Logic Execution ---
inputs = None
analysis_data = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        row = df.iloc[0]
        inputs = (row['hr'], row['temp'], row['spo2'], row['wbc'], row['creatinine'])
    except Exception:
        st.sidebar.error("CSV Error. Use columns: hr, temp, spo2, wbc, creatinine")
elif submit:
    inputs = (m_hr, m_temp, m_spo2, m_wbc, m_creatinine)

if inputs:
    analysis_data = calculate_weighted_risk(*inputs)

# --- 5. Main Dashboard (Requirement 4 & 15) ---
st.title("Biomedical Data Analytics: ICU Monitoring & Comparison")
st.markdown("---")

if inputs and analysis_data:
    hr, temp, spo2, wbc, creatinine = inputs
    risk_pct, status, color, advice, pt_breakdown = analysis_data

    # Multi-Model Comparison Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Weighted Score", "Threshold Logic", "Coefficient Model", "📊 Comparison Summary"])

    with tab1:
        st.subheader("Primary Model: Clinical Weighted Points")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Risk Percentage", f"{risk_pct:.1f}%")
            st.markdown(f"### Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        with col_b:
            st.info(f"**Recommendation:** {advice}")
        
        # New Calculation Summary Box
        st.markdown("#### 🧮 How this was calculated:")
        list_items = "".join([f"<li><b>{m}:</b> {p} point(s)</li>" for m, p in pt_breakdown.items()])
        st.markdown(f"""
            <div class="calc-box">
                <ul>{list_items}</ul>
                <h3 style="text-align: right; color: #556b2f;">Total Points: {sum(pt_breakdown.values())} / 10</h3>
            </div>
        """, unsafe_allow_html=True)

    with tab2:
        r2, n2 = threshold_binary_model(*inputs)
        st.subheader(f"Model: {n2}")
        st.metric("Predicted Risk", f"{r2:.1f}%")
        st.warning("This model triggers a high risk level if any single vital sign crosses a critical safety limit.")

    with tab3:
        r3, n3 = linear_coefficient_model(*inputs)
        st.subheader(f"Model: {n3}")
        st.metric("Predicted Risk", f"{r3:.1f}%")
        st.success("This model applies fixed mathematical weights directly to raw input values.")

    with tab4:
        st.subheader("Model Comparison Result [cite: 22]")
        r2, _ = threshold_binary_model(*inputs)
        r3, _ = linear_coefficient_model(*inputs)
        comparison_df = pd.DataFrame({
            "Model Name": ["Weighted Score", "Threshold Binary", "Coefficient Sum"],
            "Risk Prediction (%)": [risk_pct, r2, r3]
        })
        st.table(comparison_df)
        st.bar_chart(comparison_df.set_index("Model Name"))
        avg_risk = (risk_pct + r2 + r3) / 3
        st.markdown(f"### Consensus Risk Level: **{avg_risk:.1f}%**")

    # Metric Visualization (Requirement 22)
    st.markdown("---")
    st.subheader("Metric Visualization")
    st.bar_chart(pd.DataFrame({'Metric': ['HR', 'Temp', 'SpO2', 'WBC', 'Creatinine'], 
                               'Value': [hr, temp, spo2, wbc, creatinine]}).set_index('Metric'))

else:
    st.info("Please use the sidebar to input patient data and generate the analysis comparison.")

# Clinical Reference Expandable Guide
with st.expander("ℹ️ View Reference: Clinical Risk Point System"):
    st.markdown("### Breakdown of Risk Points")
    st.write("Each metric contributes to the final risk percentage based on clinical significance.")
    
    # Defining your specific point data [cite: 9, 10]
    guide_df = pd.DataFrame([
        {"Metric": "Oxygen (SpO2)", "Condition": "Below 90%", "Points": "3 Points"},
        {"Metric": "Oxygen (SpO2)", "Condition": "90% – 93%", "Points": "1 Point"},
        {"Metric": "Heart Rate", "Condition": "Above 120 or Below 50", "Points": "2 Points"},
        {"Metric": "Heart Rate", "Condition": "100–120 or 50–60", "Points": "1 Point"},
        {"Metric": "Body Temp", "Condition": "Above 38.5°C or Below 36.0°C", "Points": "2 Points"},
        {"Metric": "Body Temp", "Condition": "37.5°C–38.5°C or 36.0°C–36.5°C", "Points": "1 Point"},
        {"Metric": "WBC Count", "Condition": "Above 15.0 or Below 4.0", "Points": "2 Points"},
        {"Metric": "WBC Count", "Condition": "11.0 – 15.0", "Points": "1 Point"},
        {"Metric": "Creatinine", "Condition": "Above 1.5 mg/dL", "Points": "1 Point"}
    ])
    st.table(guide_df)

st.caption("Deliverable: Biomedical Data Analytics Working Application | Cairo University")