"""
utils.py - Utility Functions for Data Validation and Processing
Phase 2.1: Input Validation
Phase 1.1: CSV Handling
Phase 3.2: Report Generation
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from models import RiskAssessment


# ============================================================================
# PHASE 2.1: DATA VALIDATION
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


def validate_vitals(hr: float, temp: float, spo2: float, 
                   wbc: float, creatinine: float) -> Tuple[bool, List[str]]:
    """
    Phase 2.1: Comprehensive vital signs validation
    Returns: (is_valid, list_of_error_messages)
    
    Biological ranges based on clinical literature:
    - Heart Rate: 20-250 bpm (extreme range, includes athletic bradycardia and SVT)
    - Temperature: 32-43°C (survivable range, outside = incompatible with life)
    - SpO2: 50-100% (below 50% typically incompatible with consciousness)
    - WBC: 0.1-50.0 x10^9/L (extreme leukopenia to leukemia range)
    - Creatinine: 0.1-15.0 mg/dL (normal to severe renal failure)
    """
    
    errors = []
    
    # Heart Rate Validation
    if not (20 <= hr <= 250):
        errors.append(f"❌ Heart Rate ({hr} bpm) outside biological range (20-250 bpm)")
    
    # Temperature Validation
    if not (32.0 <= temp <= 43.0):
        errors.append(f"❌ Body Temperature ({temp}°C) outside survivable range (32-43°C)")
    
    # SpO2 Validation
    if not (50 <= spo2 <= 100):
        errors.append(f"❌ Oxygen Saturation ({spo2}%) must be between 50-100%")
    
    # WBC Count Validation
    if not (0.1 <= wbc <= 50.0):
        errors.append(f"❌ WBC Count ({wbc} x10⁹/L) outside biological range (0.1-50.0)")
    
    # Creatinine Validation
    if not (0.1 <= creatinine <= 15.0):
        errors.append(f"❌ Creatinine ({creatinine} mg/dL) outside biological range (0.1-15.0)")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def check_missing_data(hr: float, temp: float, spo2: float, 
                      wbc: float, creatinine: float) -> Tuple[bool, str]:
    """
    Check for missing or null values in vital signs.
    Returns: (has_missing, warning_message)
    """
    values = [hr, temp, spo2, wbc, creatinine]
    value_names = ['Heart Rate', 'Temperature', 'SpO2', 'WBC Count', 'Creatinine']
    
    missing = []
    for val, name in zip(values, value_names):
        if pd.isna(val) or val is None:
            missing.append(name)
    
    if missing:
        warning_msg = f"⚠️ Missing vital signs: {', '.join(missing)}. Prediction reliability reduced."
        return True, warning_msg
    
    return False, ""


# ============================================================================
# PHASE 1.1: CSV HANDLING & SESSION STATE
# ============================================================================

def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Phase 1.1: Validate CSV contains required columns.
    Returns: (is_valid, error_message)
    """
    required_columns = ['hr', 'temp', 'spo2', 'wbc', 'creatinine']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        error_msg = f"❌ CSV missing required columns: {', '.join(missing_columns)}\n\n"
        error_msg += f"Required columns: {', '.join(required_columns)}\n"
        error_msg += f"Found columns: {', '.join(df.columns.tolist())}"
        return False, error_msg
    
    return True, ""


def load_patient_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Phase 1.1: Load and validate CSV file containing patient data.
    Stores in session state for multi-patient selection.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate format
        is_valid, error_msg = validate_csv_format(df)
        if not is_valid:
            st.error(error_msg)
            return None
        
        # Add patient ID if not present
        if 'patient_id' not in df.columns:
            df.insert(0, 'patient_id', [f"Patient_{i+1}" for i in range(len(df))])
        
        # Store in session state
        st.session_state.patient_data = df
        st.session_state.data_loaded = True
        
        return df
    
    except pd.errors.EmptyDataError:
        st.error("❌ CSV file is empty")
        return None
    except pd.errors.ParserError:
        st.error("❌ CSV file is malformed. Please check formatting.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return None


def get_patient_by_index(df: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Phase 1.1: Extract patient data by row index.
    Returns dictionary with vital signs.
    """
    try:
        row = df.iloc[index]
        return {
            'patient_id': row.get('patient_id', f'Patient_{index+1}'),
            'hr': float(row['hr']),
            'temp': float(row['temp']),
            'spo2': float(row['spo2']),
            'wbc': float(row['wbc']),
            'creatinine': float(row['creatinine'])
        }
    except (KeyError, ValueError, IndexError) as e:
        st.error(f"❌ Error extracting patient data: {str(e)}")
        return None


# ============================================================================
# PHASE 3.2: CLINICAL REPORT GENERATION
# ============================================================================

def generate_clinical_report(patient_id: str, 
                            vitals: Dict[str, float],
                            assessment: RiskAssessment,
                            model_name: str) -> str:
    """
    Phase 3.2: Generate downloadable clinical report in text format.
    
    Args:
        patient_id: Patient identifier
        vitals: Dictionary with hr, temp, spo2, wbc, creatinine
        assessment: RiskAssessment object from model
        model_name: Name of the prediction model used
    
    Returns:
        Formatted clinical report as string
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║          ICU RISK ANALYTICS - CLINICAL ASSESSMENT REPORT             ║
║                     Cairo University Medical System                  ║
╚══════════════════════════════════════════════════════════════════════╝

REPORT METADATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient ID:          {patient_id}
Assessment Time:     {timestamp}
Analysis Model:      {model_name}
Report Version:      v1.2


VITAL SIGNS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Heart Rate:          {vitals['hr']:.1f} bpm
Body Temperature:    {vitals['temp']:.1f} °C
Oxygen Saturation:   {vitals['spo2']:.1f} %
WBC Count:           {vitals['wbc']:.2f} x10⁹/L
Creatinine:          {vitals['creatinine']:.2f} mg/dL


RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Risk Percentage:     {assessment.risk_percentage:.1f}%
Clinical Status:     {assessment.status}
Confidence Level:    {assessment.confidence:.1f}%


METRIC CONTRIBUTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for metric, points in assessment.metric_contributions.items():
        report += f"{metric:<20} {points} point(s)\n"
    
    total_points = sum(assessment.metric_contributions.values())
    report += f"{'─' * 40}\n"
    report += f"{'Total Score':<20} {total_points} points\n"
    
    report += f"""

CLINICAL RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for i, recommendation in enumerate(assessment.recommendations, 1):
        # Remove emoji for text report
        clean_rec = recommendation.encode('ascii', 'ignore').decode('ascii').strip()
        if clean_rec:
            report += f"{i}. {clean_rec}\n"
    
    report += f"""

CLINICAL NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This automated risk assessment is intended to support clinical decision-
making and should not replace professional medical judgment. All critical
alerts require immediate physician review and validation.


DATASET REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Machine Learning models trained on MIMIC-III Clinical Database:
Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database.
PhysioNet. https://doi.org/10.13026/C2XW26


AUTHORIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyzed by:         ICU Risk Analytics System v1.2
Generated:           {timestamp}
Signature:           [Electronic Signature - System Automated]


╔══════════════════════════════════════════════════════════════════════╗
║                        END OF CLINICAL REPORT                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    return report


def create_batch_export(df: pd.DataFrame, all_assessments: List[Dict]) -> pd.DataFrame:
    """
    Phase 3.2: Create exportable DataFrame with all patient results.
    
    Args:
        df: Original patient dataframe
        all_assessments: List of dictionaries containing assessment results
    
    Returns:
        DataFrame ready for CSV export
    """
    
    export_data = []
    
    for i, assessment_dict in enumerate(all_assessments):
        row_data = {
            'Patient_ID': df.iloc[i].get('patient_id', f'Patient_{i+1}'),
            'Heart_Rate': df.iloc[i]['hr'],
            'Temperature': df.iloc[i]['temp'],
            'SpO2': df.iloc[i]['spo2'],
            'WBC_Count': df.iloc[i]['wbc'],
            'Creatinine': df.iloc[i]['creatinine'],
            'Risk_Percentage': assessment_dict['risk_percentage'],
            'Clinical_Status': assessment_dict['status'],
            'Confidence': assessment_dict['confidence'],
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        export_data.append(row_data)
    
    return pd.DataFrame(export_data)


# ============================================================================
# PHASE 3.1: VISUAL RISK GAUGE GENERATION
# ============================================================================

def generate_risk_gauge_html(risk_percentage: float, color: str) -> str:
    """
    Phase 3.1: Generate HTML/CSS for visual risk gauge meter.
    
    Args:
        risk_percentage: Risk score (0-100)
        color: Color code (red, orange, green)
    
    Returns:
        HTML string for risk gauge visualization
    """
    
    # Color mapping
    color_map = {
        'red': '#dc3545',
        'orange': '#fd7e14',
        'green': '#28a745'
    }
    
    bar_color = color_map.get(color, '#6c757d')
    
    html = f"""
    <div style="
        width: 100%;
        background: linear-gradient(to right, 
            #28a745 0%, 
            #28a745 30%, 
            #fd7e14 30%, 
            #fd7e14 70%, 
            #dc3545 70%, 
            #dc3545 100%);
        height: 40px;
        border-radius: 20px;
        position: relative;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <div style="
            position: absolute;
            left: {risk_percentage}%;
            top: -10px;
            transform: translateX(-50%);
            background: {bar_color};
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 10;
        ">
            {risk_percentage:.1f}%
        </div>
        <div style="
            position: absolute;
            left: {risk_percentage}%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 4px;
            height: 60px;
            background: {bar_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        "></div>
    </div>
    <div style="
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    ">
        <span>STABLE (0-30%)</span>
        <span>MONITOR (30-70%)</span>
        <span>CRITICAL (70-100%)</span>
    </div>
    """
    
    return html


# ============================================================================
# UTILITY: SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'selected_patient_index' not in st.session_state:
        st.session_state.selected_patient_index = 0