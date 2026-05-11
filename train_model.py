"""
train_model.py - ICU Risk Model Training Pipeline
Trains a Logistic Regression model on synthetic ICU data based on MIMIC-III statistics.

For production use, replace with actual MIMIC-III data after obtaining PhysioNet credentials.

Dataset Generation Based on:
- Johnson et al. (2016) MIMIC-III Clinical Database
- Pollard et al. (2018) eICU Collaborative Research Database
- Published ICU vital sign distributions and mortality correlations

Training Target: 30-day adverse outcome (mortality/readmission)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SYNTHETIC ICU DATA GENERATION (Based on MIMIC-III Statistics)
# ============================================================================

def generate_synthetic_icu_data(n_samples=50000, random_seed=42):
    """
    Generate synthetic ICU patient data with realistic vital sign distributions
    and mortality correlations based on published MIMIC-III statistics.
    
    This function creates training data that mimics real ICU patterns when
    actual MIMIC-III access is not available.
    
    References:
    - Johnson et al. (2016) MIMIC-III Clinical Database
    - Mean vital signs and correlations from published ICU literature
    
    Args:
        n_samples: Number of synthetic patients to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with vital signs and adverse outcome labels
    """
    
    np.random.seed(random_seed)
    
    print(f"Generating {n_samples} synthetic ICU patient records...")
    print("Based on MIMIC-III published vital sign distributions")
    
    # ========================================================================
    # VITAL SIGN DISTRIBUTIONS (from MIMIC-III literature)
    # ========================================================================
    
    # Heart Rate: Mean ~85 bpm, SD ~18
    # Abnormal values (tachycardia/bradycardia) correlate with mortality
    hr_normal = np.random.normal(85, 18, int(n_samples * 0.7))
    hr_abnormal = np.concatenate([
        np.random.normal(120, 15, int(n_samples * 0.15)),  # Tachycardia
        np.random.normal(50, 10, int(n_samples * 0.15))    # Bradycardia
    ])
    hr = np.concatenate([hr_normal, hr_abnormal])
    hr = np.clip(hr, 30, 200)
    
    # Temperature: Mean ~37.0°C, SD ~1.2
    # Fever/hypothermia correlate with infection and mortality
    temp_normal = np.random.normal(37.0, 0.8, int(n_samples * 0.75))
    temp_abnormal = np.concatenate([
        np.random.normal(38.8, 0.7, int(n_samples * 0.15)),  # Fever
        np.random.normal(35.8, 0.6, int(n_samples * 0.10))   # Hypothermia
    ])
    temp = np.concatenate([temp_normal, temp_abnormal])
    temp = np.clip(temp, 34.0, 41.0)
    
    # SpO2: Mean ~96%, SD ~4
    # Hypoxemia is a strong predictor of adverse outcomes
    spo2_normal = np.random.normal(97, 2, int(n_samples * 0.75))
    spo2_low = np.random.normal(88, 5, int(n_samples * 0.25))
    spo2 = np.concatenate([spo2_normal, spo2_low])
    spo2 = np.clip(spo2, 70, 100)
    
    # WBC Count: Mean ~11.0, SD ~4.5
    # Leukocytosis/leukopenia indicate infection or immune dysfunction
    wbc_normal = np.random.normal(9.5, 2.5, int(n_samples * 0.65))
    wbc_high = np.random.normal(16.0, 4.0, int(n_samples * 0.25))   # Infection
    wbc_low = np.random.normal(3.0, 1.0, int(n_samples * 0.10))     # Immune suppression
    wbc = np.concatenate([wbc_normal, wbc_high, wbc_low])
    wbc = np.clip(wbc, 1.0, 30.0)
    
    # Creatinine: Mean ~1.1, SD ~0.9
    # Elevated creatinine indicates acute kidney injury
    creatinine_normal = np.random.normal(1.0, 0.3, int(n_samples * 0.70))
    creatinine_elevated = np.random.gamma(3, 0.8, int(n_samples * 0.30))
    creatinine = np.concatenate([creatinine_normal, creatinine_elevated])
    creatinine = np.clip(creatinine, 0.4, 8.0)
    
    # Shuffle to remove generation order bias
    indices = np.random.permutation(n_samples)
    hr = hr[indices]
    temp = temp[indices]
    spo2 = spo2[indices]
    wbc = wbc[indices]
    creatinine = creatinine[indices]
    
    # ========================================================================
    # OUTCOME GENERATION (Based on Clinical Risk Factors)
    # ========================================================================
    
    # Calculate risk score based on clinical evidence
    risk_score = np.zeros(n_samples)
    
    # Heart Rate contribution (U-shaped risk: both high and low are bad)
    risk_score += np.abs(hr - 80) * 0.02
    
    # Temperature contribution (fever/hypothermia both increase risk)
    risk_score += np.abs(temp - 37.0) * 0.8
    
    # SpO2 contribution (hypoxemia is very high risk)
    risk_score += (100 - spo2) * 0.15
    
    # WBC contribution (both high and low are concerning)
    risk_score += np.abs(wbc - 9.0) * 0.12
    
    # Creatinine contribution (kidney dysfunction)
    risk_score += creatinine * 0.6
    
    # Add some random noise (not all high-risk patients die, not all low-risk survive)
    risk_score += np.random.normal(0, 2, n_samples)
    
    # Convert risk score to probability using logistic function
    probability = 1 / (1 + np.exp(-0.3 * (risk_score - 10)))
    
    # Generate binary outcomes (30-day adverse outcome: mortality/readmission)
    # Baseline mortality rate ~15% (realistic for ICU)
    outcome = (np.random.random(n_samples) < probability).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'hr': hr,
        'temp': temp,
        'spo2': spo2,
        'wbc': wbc,
        'creatinine': creatinine,
        'adverse_outcome': outcome
    })
    
    # Add patient IDs
    data.insert(0, 'patient_id', [f'ICU_{i:06d}' for i in range(n_samples)])
    
    print(f"✓ Generated {n_samples} patient records")
    print(f"✓ Adverse outcome rate: {outcome.mean()*100:.2f}%")
    print(f"✓ Feature correlations with outcome established")
    
    return data


# ============================================================================
# MODEL TRAINING PIPELINE
# ============================================================================

def train_icu_risk_model(data, test_size=0.2, random_seed=42):
    """
    Train Logistic Regression model for ICU risk prediction.
    
    Args:
        data: DataFrame with vital signs and outcomes
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (trained_model, scaler, metrics_dict)
    """
    
    print("\n" + "="*70)
    print("TRAINING ICU RISK PREDICTION MODEL")
    print("="*70)
    
    # ========================================================================
    # 1. PREPARE FEATURES AND TARGET
    # ========================================================================
    
    feature_cols = ['hr', 'temp', 'spo2', 'wbc', 'creatinine']
    X = data[feature_cols].values
    y = data['adverse_outcome'].values
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total patients: {len(data)}")
    print(f"   Features: {', '.join(feature_cols)}")
    print(f"   Target: 30-day adverse outcome (mortality/readmission)")
    print(f"   Positive cases: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"   Negative cases: {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)")
    
    # ========================================================================
    # 2. TRAIN-TEST SPLIT
    # ========================================================================
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=y  # Maintain outcome distribution
    )
    
    print(f"\n📈 Data Split:")
    print(f"   Training set: {len(X_train)} patients")
    print(f"   Test set: {len(X_test)} patients")
    
    # ========================================================================
    # 3. FEATURE SCALING
    # ========================================================================
    
    print(f"\n⚙️ Feature Standardization:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print scaling parameters
    print(f"   Feature means: {scaler.mean_}")
    print(f"   Feature stds: {scaler.scale_}")
    
    # ========================================================================
    # 4. MODEL TRAINING
    # ========================================================================
    
    print(f"\n🤖 Training Logistic Regression Model...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        class_weight='balanced',  # Handle class imbalance
        solver='lbfgs',
        C=1.0  # Regularization strength
    )
    
    model.fit(X_train_scaled, y_train)
    
    print(f"✓ Model training complete")
    print(f"✓ Iterations: {model.n_iter_[0]}")
    print(f"✓ Coefficients learned for {len(feature_cols)} features")
    
    # Print learned coefficients
    print(f"\n📊 Learned Feature Coefficients:")
    for feature, coef in zip(feature_cols, model.coef_[0]):
        print(f"   {feature:12s}: {coef:+.4f}")
    print(f"   {'Intercept':12s}: {model.intercept_[0]:+.4f}")
    
    # ========================================================================
    # 5. MODEL EVALUATION
    # ========================================================================
    
    print(f"\n📈 Model Evaluation:")
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n   Training Set Performance:")
    print(f"      Accuracy:  {train_accuracy:.4f}")
    print(f"      Precision: {train_precision:.4f}")
    print(f"      Recall:    {train_recall:.4f}")
    print(f"      AUC-ROC:   {train_auc:.4f}")
    
    print(f"\n   Test Set Performance:")
    print(f"      Accuracy:  {test_accuracy:.4f} ⭐")
    print(f"      Precision: {test_precision:.4f} ⭐")
    print(f"      Recall:    {test_recall:.4f} ⭐")
    print(f"      AUC-ROC:   {test_auc:.4f} ⭐")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n   Confusion Matrix (Test Set):")
    print(f"      True Negatives:  {cm[0,0]:5d}   False Positives: {cm[0,1]:5d}")
    print(f"      False Negatives: {cm[1,0]:5d}   True Positives:  {cm[1,1]:5d}")
    
    # Classification Report
    print(f"\n   Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Survived', 'Adverse Outcome']))
    
    # ========================================================================
    # 6. PACKAGE METRICS
    # ========================================================================
    
    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_auc_roc': train_auc,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc_roc': test_auc,
        'feature_names': feature_cols,
        'coefficients': model.coef_[0].tolist(),
        'intercept': model.intercept_[0],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return model, scaler, metrics


# ============================================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================================

def save_trained_model(model, scaler, metrics, output_dir='.'):
    """
    Save trained model, scaler, and metadata to disk.
    
    Args:
        model: Trained LogisticRegression model
        scaler: Fitted StandardScaler
        metrics: Dictionary of performance metrics
        output_dir: Directory to save files
    """
    
    print(f"\n💾 Saving Model Artifacts...")
    
    # Save model
    model_path = f"{output_dir}/icu_risk_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = f"{output_dir}/icu_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved: {scaler_path}")
    
    # Save metrics
    metrics_path = f"{output_dir}/training_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"   ✓ Metrics saved: {metrics_path}")
    
    # Save human-readable report
    report_path = f"{output_dir}/training_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ICU RISK PREDICTION MODEL - TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write("Data Source: Synthetic ICU data based on MIMIC-III statistics\n")
        f.write(f"Training Samples: {metrics['n_train_samples']}\n")
        f.write(f"Test Samples: {metrics['n_test_samples']}\n")
        f.write(f"Features: {', '.join(metrics['feature_names'])}\n")
        f.write(f"Target: 30-day adverse outcome (mortality/readmission)\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write("Algorithm: Logistic Regression\n")
        f.write("Regularization: L2 (Ridge)\n")
        f.write("Class Weighting: Balanced\n")
        f.write(f"Intercept: {metrics['intercept']:.4f}\n\n")
        
        f.write("FEATURE COEFFICIENTS\n")
        f.write("-" * 70 + "\n")
        for feature, coef in zip(metrics['feature_names'], metrics['coefficients']):
            f.write(f"{feature:15s}: {coef:+.4f}\n")
        f.write("\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write("Training Set:\n")
        f.write(f"  Accuracy:  {metrics['train_accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['train_precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['train_recall']:.4f}\n")
        f.write(f"  AUC-ROC:   {metrics['train_auc_roc']:.4f}\n\n")
        
        f.write("Test Set:\n")
        f.write(f"  Accuracy:  {metrics['test_accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['test_precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['test_recall']:.4f}\n")
        f.write(f"  AUC-ROC:   {metrics['test_auc_roc']:.4f}\n\n")
        
        f.write("TRAINING METADATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training Date: {metrics['training_date']}\n")
        f.write("Model Version: 1.0\n")
        f.write("Framework: scikit-learn\n\n")
        
        f.write("CLINICAL INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("This model predicts 30-day adverse outcomes (mortality or readmission)\n")
        f.write("for ICU patients based on five vital sign measurements. The model was\n")
        f.write("trained on synthetic data mimicking MIMIC-III vital sign distributions\n")
        f.write("and clinical outcome correlations from published literature.\n\n")
        
        f.write("For production deployment, retrain on actual MIMIC-III data after\n")
        f.write("obtaining PhysioNet credentials at: https://mimic.mit.edu/\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"   ✓ Training report saved: {report_path}")
    
    print(f"\n✅ All artifacts saved successfully!")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline execution.
    """
    
    print("\n" + "="*70)
    print("ICU RISK PREDICTION MODEL - TRAINING PIPELINE")
    print("Cairo University - Biomedical Data Analytics Project")
    print("="*70 + "\n")
    
    # Step 1: Generate training data
    print("STEP 1: DATA GENERATION")
    print("-" * 70)
    data = generate_synthetic_icu_data(n_samples=50000, random_seed=42)
    
    # Save raw data for reference
    data.to_csv('training_data.csv', index=False)
    print(f"✓ Training data saved: training_data.csv")
    
    # Step 2: Train model
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("-" * 70)
    model, scaler, metrics = train_icu_risk_model(data, test_size=0.2, random_seed=42)
    
    # Step 3: Save artifacts
    print("\n" + "="*70)
    print("STEP 3: SAVE MODEL ARTIFACTS")
    print("-" * 70)
    save_trained_model(model, scaler, metrics, output_dir='.')
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\n✅ Model training successful!")
    print(f"✅ Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"✅ Test AUC-ROC: {metrics['test_auc_roc']:.4f}")
    print("\nGenerated Files:")
    print("   1. icu_risk_model.pkl      - Trained model")
    print("   2. icu_scaler.pkl          - Feature scaler")
    print("   3. training_metrics.pkl    - Performance metrics")
    print("   4. training_report.txt     - Human-readable report")
    print("   5. training_data.csv       - Training dataset")
    
    print("\n📝 Next Steps:")
    print("   1. Review training_report.txt for detailed metrics")
    print("   2. Update models.py to load icu_risk_model.pkl")
    print("   3. Run the Streamlit app: streamlit run app.py")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()