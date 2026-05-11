# 🚀 Quick Start Guide

Get the ICU Risk Analytics Platform running in 5 minutes!

---

## ⚡ Fast Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
**Wait time**: ~30 seconds  
**Output**: 5 files (model, scaler, metrics, report, data)

### 3. Launch the App
```bash
streamlit run app.py
```
**Access**: Opens automatically at `http://localhost:8501`

---

## 📝 First Analysis

### Option A: Use Sample Data
1. Upload `CSV_Upload_for_Medical_Data.csv` via sidebar
2. Select a patient from dropdown
3. View risk assessment instantly

### Option B: Manual Entry
1. Fill in the sidebar form:
   - Heart Rate: 95 bpm
   - Temperature: 38.2°C
   - SpO2: 92%
   - WBC: 14.5
   - Creatinine: 1.2
2. Click "Analyze Patient"
3. Explore the 4 tabs

---

## 🎯 Key Features to Try

1. **Risk Gauge**: Visual color-coded risk meter
2. **Model Comparison**: See how 3 models agree/disagree
3. **Clinical Recommendations**: Specific action items
4. **Download Report**: Get a text-based clinical summary
5. **Batch Export**: Analyze multiple patients at once

---

## ❓ Troubleshooting

### "Model file not found"
→ Run `python train_model.py` first

### "CSV Error: Use columns..."
→ Ensure CSV has: `hr,temp,spo2,wbc,creatinine`

### "Validation Failed"
→ Check values are in biological ranges:
- HR: 20-250 bpm
- Temp: 32-43°C
- SpO2: 50-100%
- WBC: 0.1-50
- Creatinine: 0.1-15

---

## 📚 Learn More

See [README.md](README.md) for:
- Detailed architecture
- Model training methodology
- Dataset information
- Clinical interpretation guide

---

**Need help?** Check the Clinical Reference Guide in the app (bottom expandable section)
