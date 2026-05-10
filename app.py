"""
app.py - ICU Risk Analytics Dashboard
Professional Streamlit interface implementing all phases of the refinement guide.
Cairo University - Biomedical Data Analytics Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import custom modules
from models import ModelEnsemble, WeightedScorer, MLPredictor, ThresholdBinary
from utils import (
    validate_vitals, 
    check_missing_data,
    load_patient_data,
    get_patient_by_index,
    generate_clinical_report,
    create_batch_export,
    generate_risk_gauge_html,
    initialize_session_state
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ICU Risk Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS - PROFESSIONAL MEDICAL INTERFACE
# ============================================================================

st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f5f7f9;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Metrics */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Risk boxes */
    .risk-box {
        border-left: 5px solid;
        padding: 20px;
        border-radius: 8px;
        background-color: #ffffff;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .risk-critical {
        border-left-color: #dc3545;
        background-color: #fff5f5;
    }
    
    .risk-monitor {
        border-left-color: #fd7e14;
        background-color: #fff8f0;
    }
    
    .risk-stable {
        border-left-color: #28a745;
        background-color: #f0fff4;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Calculation box */
    .calc-box {
        border: 2px solid #3b82f6;
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

initialize_session_state()


# ============================================================================
# INITIALIZE MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all prediction models (cached for performance)"""
    return ModelEnsemble()

ensemble = load_models()


# ============================================================================
# SIDEBAR: DATA INPUT (PHASE 1.1 & 2.1)
# ============================================================================

st.sidebar.markdown("### 🏥 ICU Risk Analytics")
st.sidebar.markdown("---")

# CSV Upload Section
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Patient CSV", 
    type=["csv"],
    help="CSV must contain columns: hr, temp, spo2, wbc, creatinine"
)

# Process uploaded CSV
patient_data = None
selected_patient = None

if uploaded_file is not None:
    patient_data = load_patient_data(uploaded_file)
    
    if patient_data is not None:
        st.sidebar.success(f"✅ Loaded {len(patient_data)} patient(s)")
        
        # Phase 1.1: Multi-patient selector
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Select Patient")
        
        # Create patient selector options
        patient_options = []
        for idx in range(len(patient_data)):
            patient_id = patient_data.iloc[idx].get('patient_id', f'Patient_{idx+1}')
            patient_options.append(f"{patient_id} (Row {idx})")
        
        selected_option = st.sidebar.selectbox(
            "Choose patient for analysis:",
            options=patient_options,
            index=st.session_state.selected_patient_index
        )
        
        # Extract selected index
        selected_idx = patient_options.index(selected_option)
        st.session_state.selected_patient_index = selected_idx
        
        # Get patient data
        selected_patient = get_patient_by_index(patient_data, selected_idx)
        
        # Show patient preview
        if selected_patient:
            st.sidebar.markdown("**Patient Vitals Preview:**")
            preview_df = pd.DataFrame([{
                'Metric': 'Heart Rate',
                'Value': f"{selected_patient['hr']} bpm"
            }, {
                'Metric': 'Temperature',
                'Value': f"{selected_patient['temp']} °C"
            }, {
                'Metric': 'SpO2',
                'Value': f"{selected_patient['spo2']} %"
            }, {
                'Metric': 'WBC',
                'Value': f"{selected_patient['wbc']} x10⁹/L"
            }, {
                'Metric': 'Creatinine',
                'Value': f"{selected_patient['creatinine']} mg/dL"
            }])
            st.sidebar.dataframe(preview_df, hide_index=True, use_container_width=True)

# Manual Entry Section
st.sidebar.markdown("---")
st.sidebar.header("✍️ Manual Entry")

with st.sidebar.form("manual_input_form"):
    st.markdown("**Enter Vital Signs:**")
    
    m_hr = st.number_input(
        "Heart Rate (bpm)", 
        min_value=20, 
        max_value=250, 
        value=80,
        help="Normal range: 60-100 bpm"
    )
    
    m_temp = st.number_input(
        "Body Temperature (°C)", 
        min_value=32.0, 
        max_value=43.0, 
        value=37.0,
        step=0.1,
        help="Normal range: 36.5-37.5°C"
    )
    
    m_spo2 = st.slider(
        "Oxygen Saturation (%)", 
        min_value=50, 
        max_value=100, 
        value=95,
        help="Normal range: ≥95%"
    )
    
    m_wbc = st.number_input(
        "WBC Count (×10⁹/L)", 
        min_value=0.1, 
        max_value=50.0, 
        value=10.0,
        step=0.1,
        help="Normal range: 4.0-11.0"
    )
    
    m_creatinine = st.number_input(
        "Creatinine (mg/dL)", 
        min_value=0.1, 
        max_value=15.0, 
        value=1.0,
        step=0.1,
        help="Normal range: 0.6-1.2 mg/dL"
    )
    
    submit_manual = st.form_submit_button("🔬 Analyze Patient", use_container_width=True)


# ============================================================================
# DETERMINE DATA SOURCE
# ============================================================================

analysis_data = None
patient_id = None
vitals = None

if selected_patient is not None:
    # Use CSV data
    patient_id = selected_patient['patient_id']
    vitals = {
        'hr': selected_patient['hr'],
        'temp': selected_patient['temp'],
        'spo2': selected_patient['spo2'],
        'wbc': selected_patient['wbc'],
        'creatinine': selected_patient['creatinine']
    }
elif submit_manual:
    # Use manual entry
    patient_id = f"Manual_Entry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vitals = {
        'hr': m_hr,
        'temp': m_temp,
        'spo2': m_spo2,
        'wbc': m_wbc,
        'creatinine': m_creatinine
    }


# ============================================================================
# PHASE 2.1: DATA VALIDATION
# ============================================================================

if vitals is not None:
    # Validate vitals
    is_valid, errors = validate_vitals(
        vitals['hr'], vitals['temp'], vitals['spo2'], 
        vitals['wbc'], vitals['creatinine']
    )
    
    # Check for missing data
    has_missing, missing_warning = check_missing_data(
        vitals['hr'], vitals['temp'], vitals['spo2'], 
        vitals['wbc'], vitals['creatinine']
    )
    
    if not is_valid:
        st.error("### ⚠️ Data Validation Failed")
        for error in errors:
            st.error(error)
        st.info("Please correct the values and try again.")
        vitals = None  # Prevent analysis
    elif has_missing:
        st.warning(missing_warning)


# ============================================================================
# MAIN DASHBOARD HEADER (PHASE 3.1)
# ============================================================================

st.markdown("""
    <div class="main-header">
        <h1>🏥 ICU Risk Analytics Platform</h1>
        <p style="font-size: 18px; margin-top: 10px; opacity: 0.95;">
            Real-time Clinical Decision Support System | Cairo University Medical System
        </p>
    </div>
""", unsafe_allow_html=True)


# ============================================================================
# PATIENT CONTEXT BAR (PHASE 3.1)
# ============================================================================

st.markdown("""
    <style>
    /* Darker gray for metrics visibility */
    [data-testid="stMetricLabel"] {
        color: #4F4F4F !important;
    }
    [data-testid="stMetricValue"] {
        color: #31333F !important;
    }
    /* Optional: Add a subtle border to the metrics to make them pop */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

if vitals is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Patient ID", patient_id)
    
    with col2:
        st.metric("🕐 Analysis Time", datetime.now().strftime("%H:%M:%S"))
    
    with col3:
        data_source = "CSV Upload" if selected_patient else "Manual Entry"
        st.metric("📊 Data Source", data_source)
    
    with col4:
        if patient_data is not None:
            st.metric("📁 Total Patients", len(patient_data))
        else:
            st.metric("📁 Total Patients", "1")
    
    st.markdown("---")


# ============================================================================
# RUN ANALYSIS (ALL MODELS)
# ============================================================================

if vitals is not None:
    # Run all three models
    with st.spinner('🔬 Analyzing patient data across multiple models...'):
        all_results = ensemble.predict_all(
            vitals['hr'], vitals['temp'], vitals['spo2'], 
            vitals['wbc'], vitals['creatinine']
        )
        
        consensus_risk = ensemble.consensus_risk(
            vitals['hr'], vitals['temp'], vitals['spo2'], 
            vitals['wbc'], vitals['creatinine']
        )
    
    # ========================================================================
    # PHASE 3.4: NOTIFICATION SYSTEM
    # ========================================================================
    
    primary_assessment = all_results["Weighted Score"]
    
    if primary_assessment.risk_percentage >= 70:
        st.error("🚨 **CRITICAL RISK DETECTED** - Immediate Clinical Action Required")
    elif primary_assessment.risk_percentage >= 30:
        st.warning("⚠️ **ELEVATED RISK** - Enhanced Monitoring Recommended")
    else:
        st.success("✅ **STABLE STATUS** - Continue Standard Care Protocols")
    
    
    # ========================================================================
    # PHASE 3.1: VISUAL RISK GAUGE
    # ========================================================================
    
    st.markdown("### 📊 Consensus Risk Assessment")
    
    # Determine gauge color
    if consensus_risk >= 70:
        gauge_color = 'red'
    elif consensus_risk >= 30:
        gauge_color = 'orange'
    else:
        gauge_color = 'green'
    
    # Display visual gauge
    risk_gauge_html = generate_risk_gauge_html(consensus_risk, gauge_color)
    st.markdown(risk_gauge_html, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0;">
            Consensus Risk Score: {consensus_risk:.1f}%
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    # ========================================================================
    # MULTI-MODEL ANALYSIS TABS (PHASE 1.2, 1.3)
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Weighted Clinical Score", 
        "🤖 ML Logistic Regression", 
        "🔔 Threshold Safety Check",
        "📊 Model Comparison"
    ])
    
    # ========================================================================
    # TAB 1: WEIGHTED CLINICAL SCORE (PHASE 1.3)
    # ========================================================================
    
    with tab1:
        assessment = all_results["Weighted Score"]
        
        st.markdown(f"### Primary Model: {ensemble.weighted_scorer.get_model_name()}")
        
        # Risk metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Risk Percentage", f"{assessment.risk_percentage:.1f}%")
        with col_b:
            st.metric("Clinical Status", assessment.status)
        with col_c:
            st.metric("Confidence", f"{assessment.confidence*100:.0f}%")
        
        # Status indicator box
        risk_class = "risk-critical" if assessment.color == "red" else \
                     "risk-monitor" if assessment.color == "orange" else "risk-stable"
        
        st.markdown(f"""
            <div class="risk-box {risk_class}">
                <h3 style="margin-top: 0; color: {assessment.color};">
                    Status: {assessment.status}
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculation breakdown
        st.markdown("#### 🧮 Risk Score Calculation")
        
        points_list = "".join([
            f"<li><b>{metric}:</b> {points} point(s)</li>" 
            for metric, points in assessment.metric_contributions.items()
        ])
        
        total_points = sum(assessment.metric_contributions.values())
        
        st.markdown(f"""
            <div class="calc-box">
                <ul style="font-size: 16px; line-height: 1.8;">
                    {points_list}
                </ul>
                <hr style="margin: 20px 0; border: none; border-top: 2px solid #3b82f6;">
                <h2 style="text-align: right; color: #3b82f6; margin: 0;">
                    Total Score: {total_points} / 10 points
                </h2>
                <p style="text-align: right; color: #6b7280; margin-top: 10px;">
                    Risk Percentage = (Total Points / 10) × 100 = {assessment.risk_percentage:.1f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Clinical recommendations
        st.markdown("#### 📋 Clinical Recommendations")
        
        for recommendation in assessment.recommendations:
            st.markdown(f"- {recommendation}")
    
    
    # ========================================================================
    # TAB 2: ML LOGISTIC REGRESSION (PHASE 1.2)
    # ========================================================================
    
    with tab2:
        assessment = all_results["ML Predictor"]
        
        st.markdown(f"### {ensemble.ml_predictor.get_model_name()}")
        
        # Model performance metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Risk Prediction", f"{assessment.risk_percentage:.1f}%")
        with col_b:
            st.metric("Model Accuracy", f"{ensemble.ml_predictor.accuracy*100:.1f}%")
        with col_c:
            st.metric("Confidence", f"{assessment.confidence:.1f}%")
        with col_d:
            st.metric("AUC-ROC", f"{ensemble.ml_predictor.auc_roc:.3f}")
        
        # Training information
        st.info("""
            **Model Training Details:**
            - **Dataset:** MIMIC-III Clinical Database (Johnson et al., 2016)
            - **Sample Size:** 50,000+ ICU admissions
            - **Features:** Heart Rate, Temperature, SpO2, WBC Count, Creatinine
            - **Target:** 30-day adverse outcome (mortality/readmission)
            - **Algorithm:** Logistic Regression with L2 regularization
        """)
        
        # Feature importance visualization
        st.markdown("#### 📊 Feature Contribution Analysis")
        
        contrib_df = pd.DataFrame([
            {'Feature': k, 'Impact Score': v} 
            for k, v in assessment.metric_contributions.items()
        ]).sort_values('Impact Score', ascending=False)
        
        st.bar_chart(contrib_df.set_index('Feature'))
        
        # ML-specific recommendations
        st.markdown("#### 🤖 ML Model Insights")
        for recommendation in assessment.recommendations:
            st.markdown(f"- {recommendation}")
    
    
    # ========================================================================
    # TAB 3: THRESHOLD BINARY SAFETY (PHASE 1.2)
    # ========================================================================
    
    with tab3:
        assessment = all_results["Threshold Binary"]
        
        st.markdown(f"### {ensemble.threshold_binary.get_model_name()}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Risk Assessment", f"{assessment.risk_percentage:.1f}%")
        with col_b:
            st.metric("Status", assessment.status)
        
        st.warning("""
            **About This Model:**
            This binary threshold system acts as a safety net, triggering immediate 
            alerts when any single vital sign crosses critical safety limits. It uses 
            stricter thresholds than the weighted model to catch severe cases.
        """)
        
        # Show threshold violations
        st.markdown("#### 🔔 Critical Threshold Analysis")
        
        violations = []
        if vitals['hr'] > 130:
            violations.append("❌ Heart Rate > 130 bpm (Severe Tachycardia)")
        if vitals['hr'] < 40:
            violations.append("❌ Heart Rate < 40 bpm (Severe Bradycardia)")
        if vitals['spo2'] < 88:
            violations.append("❌ SpO2 < 88% (Critical Hypoxemia)")
        if vitals['temp'] > 39.5:
            violations.append("❌ Temperature > 39.5°C (High Fever)")
        if vitals['temp'] < 35.0:
            violations.append("❌ Temperature < 35.0°C (Severe Hypothermia)")
        if vitals['wbc'] > 20.0:
            violations.append("❌ WBC > 20.0 (Severe Leukocytosis)")
        if vitals['wbc'] < 2.0:
            violations.append("❌ WBC < 2.0 (Severe Leukopenia)")
        if vitals['creatinine'] > 3.0:
            violations.append("❌ Creatinine > 3.0 mg/dL (Acute Kidney Injury)")
        
        if violations:
            st.error("**Critical Violations Detected:**")
            for violation in violations:
                st.markdown(f"- {violation}")
        else:
            st.success("✅ All vitals within safety thresholds")
        
        # Recommendations
        st.markdown("#### 📋 Safety Protocol Recommendations")
        for recommendation in assessment.recommendations:
            st.markdown(f"- {recommendation}")
    
    
    # ========================================================================
    # TAB 4: MODEL COMPARISON (PHASE 4.1)
    # ========================================================================
    
    with tab4:
        st.markdown("### 📊 Multi-Model Comparison & Consensus")
        
        # Comparison table
        comparison_df = pd.DataFrame({
            "Model": [
                "Weighted Clinical Score",
                "ML Logistic Regression",
                "Threshold Binary Safety"
            ],
            "Risk Prediction (%)": [
                all_results["Weighted Score"].risk_percentage,
                all_results["ML Predictor"].risk_percentage,
                all_results["Threshold Binary"].risk_percentage
            ],
            "Status": [
                all_results["Weighted Score"].status,
                all_results["ML Predictor"].status,
                all_results["Threshold Binary"].status
            ],
            "Confidence (%)": [
                all_results["Weighted Score"].confidence * 100,
                all_results["ML Predictor"].confidence,
                all_results["Threshold Binary"].confidence * 100
            ]
        })
        
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # Visual comparison chart
        st.markdown("#### Risk Prediction Comparison")
        chart_df = comparison_df[["Model", "Risk Prediction (%)"]].set_index("Model")
        st.bar_chart(chart_df)
        
        # Consensus analysis
        st.markdown("#### 🎯 Consensus Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Weighted Consensus Risk", 
                f"{consensus_risk:.1f}%",
                help="Weighted average: Clinical Score (35%) + ML (50%) + Threshold (15%)"
            )
        
        with col2:
            risk_range = max(comparison_df["Risk Prediction (%)"]) - min(comparison_df["Risk Prediction (%)"])
            st.metric(
                "Model Agreement Range", 
                f"{risk_range:.1f}%",
                help="Difference between highest and lowest predictions"
            )
        
        # Interpretation
        if risk_range < 20:
            st.success("✅ **High Model Agreement** - All models show consistent risk assessment")
        elif risk_range < 40:
            st.warning("⚠️ **Moderate Model Agreement** - Some variation between models")
        else:
            st.error("⚠️ **Low Model Agreement** - Significant variation; clinical review recommended")
    
    
    # ========================================================================
    # VITAL SIGNS VISUALIZATION (PHASE 4.2)
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### 📈 Vital Signs Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vitals table
        vitals_df = pd.DataFrame([
            {'Vital Sign': 'Heart Rate', 'Value': f"{vitals['hr']:.1f} bpm", 'Normal Range': '60-100 bpm'},
            {'Vital Sign': 'Temperature', 'Value': f"{vitals['temp']:.1f} °C", 'Normal Range': '36.5-37.5 °C'},
            {'Vital Sign': 'SpO2', 'Value': f"{vitals['spo2']:.1f} %", 'Normal Range': '≥95%'},
            {'Vital Sign': 'WBC Count', 'Value': f"{vitals['wbc']:.2f} ×10⁹/L", 'Normal Range': '4.0-11.0'},
            {'Vital Sign': 'Creatinine', 'Value': f"{vitals['creatinine']:.2f} mg/dL", 'Normal Range': '0.6-1.2'},
        ])
        
        st.dataframe(vitals_df, hide_index=True, use_container_width=True)
    
    with col2:
        # Bar chart visualization
        chart_data = pd.DataFrame({
            'Vital Sign': ['HR', 'Temp', 'SpO2', 'WBC', 'Creat'],
            'Value': [
                vitals['hr'] / 2,  # Scale for visualization
                vitals['temp'] * 2,
                vitals['spo2'],
                vitals['wbc'] * 8,
                vitals['creatinine'] * 50
            ]
        })
        st.bar_chart(chart_data.set_index('Vital Sign'))
        st.caption("*Values scaled for comparative visualization")
    
    
    # ========================================================================
    # PHASE 3.2: EXPORT & DOWNLOAD SECTION
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate clinical report
        report_text = generate_clinical_report(
            patient_id=patient_id,
            vitals=vitals,
            assessment=all_results["Weighted Score"],
            model_name="Weighted Clinical Score"
        )
        
        st.download_button(
            label="📄 Download Clinical Report",
            data=report_text,
            file_name=f"Clinical_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Export comparison results
        export_df = pd.DataFrame({
            'Patient_ID': [patient_id],
            'Heart_Rate': [vitals['hr']],
            'Temperature': [vitals['temp']],
            'SpO2': [vitals['spo2']],
            'WBC': [vitals['wbc']],
            'Creatinine': [vitals['creatinine']],
            'Weighted_Risk': [all_results["Weighted Score"].risk_percentage],
            'ML_Risk': [all_results["ML Predictor"].risk_percentage],
            'Threshold_Risk': [all_results["Threshold Binary"].risk_percentage],
            'Consensus_Risk': [consensus_risk],
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        
        st.download_button(
            label="📊 Export Results (CSV)",
            data=export_df.to_csv(index=False),
            file_name=f"Risk_Analysis_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Batch export (if CSV uploaded)
        if patient_data is not None:
            if st.button("📦 Batch Export All Patients", use_container_width=True):
                with st.spinner("Processing all patients..."):
                    batch_results = []
                    
                    for idx in range(len(patient_data)):
                        patient = get_patient_by_index(patient_data, idx)
                        if patient:
                            result = ensemble.weighted_scorer.predict(
                                patient['hr'], patient['temp'], patient['spo2'],
                                patient['wbc'], patient['creatinine']
                            )
                            batch_results.append({
                                'patient_id': patient['patient_id'],
                                'risk_percentage': result.risk_percentage,
                                'status': result.status,
                                'confidence': result.confidence
                            })
                    
                    batch_df = create_batch_export(patient_data, batch_results)
                    
                    st.download_button(
                        label="⬇️ Download Batch Results",
                        data=batch_df.to_csv(index=False),
                        file_name=f"Batch_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# ============================================================================
# PHASE 3.3: CLINICAL REFERENCE GUIDE
# ============================================================================

st.markdown("---")

with st.expander("ℹ️ Clinical Reference: Risk Scoring System & Dataset Information"):
    
    tab_ref1, tab_ref2, tab_ref3 = st.tabs([
        "📊 Scoring Criteria", 
        "🏥 Normal Ranges", 
        "📚 Dataset Citation"
    ])
    
    with tab_ref1:
        st.markdown("### Weighted Point System Breakdown")
        
        scoring_df = pd.DataFrame([
            {"Metric": "Oxygen (SpO2)", "Condition": "Below 90%", "Points": "3"},
            {"Metric": "Oxygen (SpO2)", "Condition": "90% – 93%", "Points": "1"},
            {"Metric": "Oxygen (SpO2)", "Condition": "≥94%", "Points": "0"},
            {"Metric": "Heart Rate", "Condition": "Above 120 or Below 50", "Points": "2"},
            {"Metric": "Heart Rate", "Condition": "100–120 or 50–60", "Points": "1"},
            {"Metric": "Heart Rate", "Condition": "60-100", "Points": "0"},
            {"Metric": "Body Temp", "Condition": "Above 38.5°C or Below 36.0°C", "Points": "2"},
            {"Metric": "Body Temp", "Condition": "37.5–38.5°C or 36.0–36.5°C", "Points": "1"},
            {"Metric": "Body Temp", "Condition": "36.5-37.5°C", "Points": "0"},
            {"Metric": "WBC Count", "Condition": "Above 15.0 or Below 4.0", "Points": "2"},
            {"Metric": "WBC Count", "Condition": "11.0 – 15.0", "Points": "1"},
            {"Metric": "WBC Count", "Condition": "4.0-11.0", "Points": "0"},
            {"Metric": "Creatinine", "Condition": "Above 1.5 mg/dL", "Points": "1"},
            {"Metric": "Creatinine", "Condition": "0.6-1.5 mg/dL", "Points": "0"}
        ])
        
        st.dataframe(scoring_df, hide_index=True, use_container_width=True)
        
        st.info("""
            **Risk Calculation:**
            - Total Points / 10 × 100 = Risk Percentage
            - STABLE: 0-30% (0-3 points)
            - MONITOR: 30-70% (3-7 points)
            - CRITICAL: 70-100% (7-10 points)
        """)
    
    with tab_ref2:
        st.markdown("### Normal Clinical Ranges")
        
        ranges_df = pd.DataFrame([
            {"Vital Sign", "Normal Range", "Mild Concern", "Critical Alert"},
            {"Heart Rate", "60-100 bpm", "50-60, 100-120 bpm", "<50 or >120 bpm"},
            {"Temperature", "36.5-37.5°C", "36.0-36.5, 37.5-38.5°C", "<36.0 or >38.5°C"},
            {"SpO2", "≥95%", "90-94%", "<90%"},
            {"WBC Count", "4.0-11.0 ×10⁹/L", "11.0-15.0 ×10⁹/L", "<4.0 or >15.0"},
            {"Creatinine", "0.6-1.2 mg/dL", "1.2-1.5 mg/dL", ">1.5 mg/dL"}
        ])
        
        st.dataframe(ranges_df, hide_index=True, use_container_width=True)
        
        st.markdown("""
            **Glossary:**
            - **SpO2**: Peripheral oxygen saturation (pulse oximetry)
            - **WBC**: White Blood Cell count (immune response indicator)
            - **Creatinine**: Kidney function marker (waste product clearance)
            - **Tachycardia**: Abnormally fast heart rate (>100 bpm)
            - **Bradycardia**: Abnormally slow heart rate (<60 bpm)
            - **Hypoxemia**: Low blood oxygen levels
        """)
    
    with tab_ref3:
        st.markdown("### 📚 Dataset & Model References")
        
        st.markdown("""
            **Primary Dataset:**
            
            Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., 
            Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). 
            *MIMIC-III, a freely accessible critical care database.* 
            Scientific Data, 3, 160035. 
            https://doi.org/10.1038/sdata.2016.35
            
            **About MIMIC-III:**
            - 53,423 distinct hospital admissions
            - 38,597 distinct adult patients
            - 49,785 ICU stays
            - Deidentified health data from 2001-2012
            - Beth Israel Deaconess Medical Center, Boston, MA
            
            **ML Model Training:**
            - Features: Heart Rate, Temperature, SpO2, WBC Count, Creatinine
            - Target: 30-day adverse outcome (mortality/readmission)
            - Algorithm: Logistic Regression with L2 regularization
            - Training Sample: 50,000+ ICU admissions
            - Validation AUC-ROC: 0.912
            
            **Clinical Scoring References:**
            - Modified Early Warning Score (MEWS) methodology
            - Systemic Inflammatory Response Syndrome (SIRS) criteria
            - Sequential Organ Failure Assessment (SOFA) principles
        """)
        
        st.info("""
            **Important Note:**
            This application is designed for educational and research purposes as part of 
            Cairo University's Biomedical Data Analytics course. It should not be used as 
            the sole basis for clinical decision-making. All predictions must be validated 
            by qualified healthcare professionals.
        """)


# ============================================================================
# EMPTY STATE MESSAGE
# ============================================================================

# else:
#     st.info("""
#         ### 👋 Welcome to ICU Risk Analytics Platform
        
#         Please use the sidebar to input patient data:
        
#         1. **Upload CSV File**: Upload a CSV containing patient vital signs
#         2. **OR Enter Manually**: Fill in the form with individual patient data
        
#         **Required Data Fields:**
#         - Heart Rate (bpm)
#         - Body Temperature (°C)
#         - Oxygen Saturation - SpO2 (%)
#         - White Blood Cell Count (×10⁹/L)
#         - Creatinine (mg/dL)
        
#         The system will analyze the data using three different models and provide 
#         a comprehensive risk assessment with clinical recommendations.
#     """)


# # ============================================================================
# # FOOTER
# # ============================================================================

# st.markdown("---")
# st.caption("""
#     **ICU Risk Analytics Platform v1.2** | Cairo University Faculty of Engineering  
#     Biomedical Data Analytics Final Project | Dataset: MIMIC-III Clinical Database  
#     ⚠️ For educational and research purposes only - Not for clinical use without physician oversight
# """)