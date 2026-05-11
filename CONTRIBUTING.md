# Contributing to ICU Risk Analytics Platform

Thank you for your interest in improving this project! This is an academic project for Cairo University, but we welcome contributions that enhance its educational value and functionality.

---

## 🎯 How to Contribute

### 1. Reporting Issues

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title describing the problem
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Screenshots if applicable
   - Your environment (Python version, OS)

### 2. Suggesting Enhancements

For feature requests:

1. **Describe the feature** and its use case
2. **Explain the benefits** for clinical decision support
3. **Consider clinical validity** - cite medical literature if possible
4. **Propose implementation** if you have technical ideas

### 3. Submitting Code

#### Setup Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/icu-risk-analytics.git
cd icu-risk-analytics

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the app
streamlit run app.py
```

#### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update docstrings

3. **Test your changes**
   - Ensure the app runs without errors
   - Test with sample CSV data
   - Verify biological range validation

4. **Commit with clear messages**
   ```bash
   git commit -am "Add feature: description of what changed"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

---

## 📋 Coding Standards

### Python Style Guide

- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use when appropriate
- **Docstrings**: Document all functions and classes
- **Comments**: Explain "why", not "what"

### Example Function

```python
def calculate_risk_score(hr: float, temp: float, spo2: float) -> float:
    """
    Calculate patient risk score from vital signs.
    
    Args:
        hr: Heart rate in beats per minute (20-250)
        temp: Body temperature in Celsius (32-43)
        spo2: Oxygen saturation percentage (50-100)
    
    Returns:
        Risk score as percentage (0-100)
    
    Raises:
        ValueError: If any vital sign is out of biological range
    """
    # Validate inputs
    if not (20 <= hr <= 250):
        raise ValueError(f"Heart rate {hr} outside valid range")
    
    # Calculate risk (example)
    risk = (abs(hr - 80) * 0.5) + ((100 - spo2) * 2.0)
    return min(risk, 100.0)
```

### Code Organization

- **models.py**: Risk prediction algorithms only
- **utils.py**: Helper functions (validation, I/O, processing)
- **app.py**: UI components and Streamlit interface only
- **train_model.py**: ML training pipeline only

### Commit Message Format

```
<type>: <short summary>

<optional body with more details>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: Add SHAP explainability for ML model

Implements SHAP values to show feature contributions
for each prediction. Helps clinicians understand why
the model made a specific prediction.

Closes #42
```

---

## 🔬 Clinical Accuracy Guidelines

### Medical Validation Required

Any changes affecting clinical logic must:

1. **Cite medical literature** supporting the change
2. **Include range justification** for thresholds
3. **Consider false positive/negative implications**
4. **Maintain conservative approach** (err on side of caution)

### Example: Adding New Vital Sign

If adding lactate measurement:

```python
# ✅ GOOD: Well-documented with clinical justification
def assess_lactate(lactate: float) -> int:
    """
    Assess lactate level for sepsis risk.
    
    Clinical thresholds based on:
    - Surviving Sepsis Campaign Guidelines (2021)
    - Lactate >2.0 mmol/L indicates tissue hypoperfusion
    - Lactate >4.0 mmol/L indicates severe sepsis
    
    Args:
        lactate: Serum lactate in mmol/L (normal: 0.5-2.0)
    
    Returns:
        Risk points (0-3)
    """
    if lactate > 4.0:
        return 3  # Severe sepsis indicator
    elif lactate > 2.0:
        return 2  # Tissue hypoperfusion
    elif lactate > 1.5:
        return 1  # Mild elevation
    else:
        return 0  # Normal range
```

```python
# ❌ BAD: No justification or citation
def assess_lactate(lactate):
    if lactate > 5:
        return 3
    elif lactate > 3:
        return 1
    return 0
```

---

## 🧪 Testing Guidelines

### Manual Testing Checklist

Before submitting a PR, test:

- [ ] App launches without errors
- [ ] CSV upload works with sample data
- [ ] Manual entry form validates inputs correctly
- [ ] All three models produce predictions
- [ ] Risk gauge displays correctly
- [ ] Reports download successfully
- [ ] Batch export works for multiple patients

### Example Test Cases

```python
# Test 1: Normal vitals should yield low risk
hr=75, temp=37.0, spo2=98, wbc=8.5, creatinine=1.0
Expected: STABLE, risk < 30%

# Test 2: Critical vitals should yield high risk
hr=140, temp=39.5, spo2=85, wbc=22.0, creatinine=3.5
Expected: CRITICAL, risk > 70%

# Test 3: Boundary values should not crash
hr=20, temp=32.0, spo2=50, wbc=0.1, creatinine=0.1
Expected: Valid calculation, no errors
```

---

## 📚 Documentation Standards

### When to Update Documentation

Update docs when you:

- Add new features
- Change user interface
- Modify validation rules
- Update ML model
- Change dataset

### Documentation Files

- **README.md**: High-level overview, installation, usage
- **QUICKSTART.md**: Fast setup for new users
- **Code comments**: Explain complex logic
- **Docstrings**: Document all public functions
- **training_report.txt**: Model performance (auto-generated)

---

## 🎓 Educational Focus

This project is primarily educational. Contributions should:

### Enhance Learning Value

- **Clear explanations** of medical concepts
- **Well-commented code** for understanding
- **Reproducible results** for verification
- **Academic citations** for credibility

### Maintain Academic Integrity

- **No plagiarism**: Original code and properly cited sources
- **Open source**: Compatible with MIT license
- **Transparent methodology**: Document all assumptions

---

## 🚫 What NOT to Contribute

Please avoid:

1. **Closed-source dependencies** that students can't access
2. **Production medical data** (HIPAA/privacy violations)
3. **Untested clinical thresholds** without citations
4. **Breaking changes** without discussion first
5. **Code without comments** or documentation

---

## 🤝 Code Review Process

### What We Look For

1. **Correctness**: Does it work as intended?
2. **Clinical validity**: Are thresholds medically sound?
3. **Code quality**: Is it readable and maintainable?
4. **Documentation**: Is it well-explained?
5. **Testing**: Has it been manually verified?

### Review Timeline

- **Simple fixes**: 1-3 days
- **New features**: 3-7 days
- **Major changes**: 7-14 days (requires discussion)

---

## 💡 Ideas for Contributions

### Beginner-Friendly

- [ ] Add more sample CSV files with diverse patient scenarios
- [ ] Improve error messages for better user guidance
- [ ] Add unit tests for validation functions
- [ ] Create video tutorial or screenshots
- [ ] Translate UI to Arabic

### Intermediate

- [ ] Add temporal trend analysis (time-series vitals)
- [ ] Implement SHAP explainability visualization
- [ ] Add more vital signs (lactate, pH, troponin)
- [ ] Create mobile-responsive design improvements
- [ ] Add patient history tracking

### Advanced

- [ ] Integrate with real MIMIC-III data
- [ ] Implement LSTM for time-series prediction
- [ ] Add multi-center validation
- [ ] Create RESTful API endpoint
- [ ] Build React Native mobile app

---

## 📞 Getting Help

### Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Create an Issue with "bug" label
- **Feature ideas**: Create an Issue with "enhancement" label
- **Security concerns**: Email directly (don't create public issue)

### Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [MIMIC-III Documentation](https://mimic.mit.edu/docs/)
- [Python Style Guide (PEP 8)](https://pep8.org/)

---

## 🏆 Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in academic presentations (if appropriate)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).

---

**Thank you for helping improve ICU Risk Analytics! 🙏**

*Together, we can build better tools for clinical decision support and medical education.*
