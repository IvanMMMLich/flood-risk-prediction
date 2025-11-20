# 🎯 Loan Approval Prediction - Subprime Lending Model

## 📊 Project Overview
**Kaggle Competition:** Loan Approval Prediction  
**Author:** Ivan Sytsev  
**Final Score:** ROC-AUC 0.9535  
**Ranking:** [Your Kaggle Rank]  
**Date:** November 2024

## 🏆 Key Results
- **Baseline Model:** 0.8740 ROC-AUC (Logistic Regression)
- **Feature Engineering:** +5.8% improvement 
- **Final Model:** 0.9535 ROC-AUC (XGBoost)
- **Total Improvement:** +8% from baseline

## 💡 Key Insights
This dataset represents a **subprime lender** with inverted risk patterns:
- Grade G (worst) → 82% approval rate
- Grade A (best) → 5% approval rate  
- High debt-to-income (>35%) → 72% approval
- Previous defaults → 3x higher approval rate

## 📁 Project Structure
```
loan-approval-prediction/
├── data/                   # Data pipeline
│   ├── raw/               # Original Kaggle data
│   ├── processed/         # Cleaned data
│   └── engineered/        # Feature engineered data
├── src/                    # Source code
│   ├── 01_eda/           # Exploratory analysis (6 steps)
│   ├── 02_preprocessing/ # Data cleaning
│   └── 03_modeling/      # Final models
├── results/               # Analysis results
│   ├── model_versions/   # Model evolution (v0-v3)
│   └── step2-5_*/       # Analysis outputs
├── submissions/          # Kaggle submission files
└── final_model.pkl      # Production-ready model
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.13+
# macOS/Linux (Windows users use WSL)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For XGBoost on macOS
brew install libomp  # Required for XGBoost
```

### Reproduce Results
```bash
# Run complete pipeline
python src/01_eda/step1_first_look.py
python src/01_eda/step2_deep_explore.py
python src/02_preprocessing/step3_check_quality.py
python src/01_eda/step4_target_analysis.py
python src/01_eda/step5_feature_importance.py
python src/01_eda/step6_feature_engineering.py
python src/03_modeling/step7_final_model.py

# Or run fast version (1 minute)
python src/03_modeling/step7_final_fast.py
```

## 📈 Model Performance

### Evolution
| Version | Model | ROC-AUC | Improvement |
|---------|-------|---------|-------------|
| v0_baseline | Logistic Regression | 0.8740 | - |
| v1_cleaned | Logistic Regression | 0.8730 | -0.1% |
| v2_engineered | Random Forest | 0.9316 | +5.8% |
| v3_final | XGBoost | 0.9535 | +8.0% |

### Feature Importance (Top 5)
1. **loan_int_rate** (14.6%) - Interest rate
2. **high_risk_score** (12.4%) - Custom feature combining 4 risk factors
3. **income_loan_ratio** (9.8%) - Income to loan amount ratio
4. **loan_grade_D** (9.6%) - Grade D indicator
5. **loan_percent_income** (9.1%) - Debt-to-income ratio

## 🔧 Feature Engineering
Created 10 intelligent features capturing subprime patterns:
- `high_risk_score`: Combination of 4 risk factors (0-4 scale)
- `is_subprime_client`: Bad grade + high DTI
- `dti_critical/high`: Debt burden indicators  
- `age_risk_young/old`: Age group risks
- `bad_grade_high_rate`: Worst grades with high rates
- And 5 more domain-specific features

## 📊 Technologies Used
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Development:** Python 3.13, VSCode, Git

## 📝 Lessons Learned
1. **Domain understanding crucial:** Recognizing subprime patterns was key
2. **Feature engineering > Complex models:** Smart features gave +5.8% boost
3. **Systematic approach wins:** Step-by-step analysis revealed insights
4. **Data quality matters:** Even 3 anomalies (0.005%) affected results

## 🎯 Business Recommendations
1. **Focus on Germany:** 37.9% churn rate vs 16% in other countries
2. **Target Grade D-G clients:** Core profitable segment  
3. **Monitor high DTI clients:** 72% approval but higher risk
4. **Cross-sell critical:** Single product clients = highest churn

## 📧 Contact
- **LinkedIn:** [Your LinkedIn]
- **GitHub:** [Your GitHub]
- **Email:** [Your Email]

## 📜 License
MIT License - feel free to use this code for learning!

---
*This project was completed as part of a Kaggle competition to demonstrate end-to-end ML pipeline development.*