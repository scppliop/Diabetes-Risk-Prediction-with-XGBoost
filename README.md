# Diabetes-Risk-Prediction-with-XGBoost
# 🏥 Diabetes Risk Prediction with XGBoost

**Analysis Date:** November 2025

---

## 1️⃣ Overview

### Background
Diabetes affects **34.2 million Americans** with annual costs approaching **$400 billion**. Early risk prediction enables:
- Targeted interventions for high-risk populations
- Cost-effective resource allocation
- Prevention through lifestyle modifications

### Methodology and Analysis Period
- **Methodology:** Machine Learning (XGBoost vs Random Forest comparison)
- **Analyst:** [Your Name]
- **Dataset:** CDC BRFSS 2015 (253,680 samples)
- **Analysis Period:** [Your dates]

### Analysis Approach
- **Data Source:** CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Techniques:**
  - 3-class Classification (0: No diabetes, 1: Prediabetes, 2: Diabetes)
  - Feature Importance Analysis
  - Conditional Risk Analysis (interaction effects)
  - Model Performance Comparison

---

## 2️⃣ Dataset

### Source
| Dataset | Features | Samples | Included in Analysis |
|---------|----------|---------|---------------------|
| **diabetes_012_health_indicators** | 21 health indicators, 3-class target | **253,680** | **✅ Included** |
| diabetes_binary_5050split | 21 health indicators, binary target | 70,692 | ❌ Excluded |
| diabetes_binary_health_indicators | 21 health indicators, binary target | 253,680 | ❌ Excluded |

**Selection Rationale:** Larger sample size (253,680) beneficial for model training and enables detailed risk differentiation (no diabetes/prediabetes/diabetes).

### Dataset Structure
```
Total Samples: 253,680
Features (21): High Blood Pressure (HighBP), High Cholesterol (HighChol), BMI, 
               General Health (GenHlth), Age, Sex, Education, Income,
               Physical Activity (PhysActivity), Fruits, Veggies, 
               Smoking, Alcohol, etc.
Target: Diabetes_012 
  ├─ 0: No diabetes or only during pregnancy
  ├─ 1: Prediabetes
  └─ 2: Diabetes
Train/Test Split: 80/20 (202,944 / 50,736)
```

### Class Distribution

| Class | Description | Count | Percentage |
|-------|------------|-------|------------|
| 0 | No diabetes | 213,703 | 84.2% |
| 1 | Prediabetes | 4,631 | 1.8% |
| 2 | Diabetes | 35,346 | 13.9% |

**Class Imbalance:** Severe imbalance exists (prediabetes class particularly small)

**Imbalance Handling:**
```python
scale_pos_weight parameter to assign weights to minority classes
Stratified sampling for train/test split
```

### Utilized Data
| Data Split | Description | Count |
|-----------|-------------|-------|
| Training Set | Model training data (80%) | 202,944 |
| Test Set | Model evaluation data (20%) | 50,736 |
| Class 0 | No diabetes | 213,703 (84.2%) |
| Class 1 | Prediabetes | 4,631 (1.8%) |
| Class 2 | Diabetes | 35,346 (13.9%) |

---

## 3️⃣ Analysis Details

### 3.1 Model Selection Analysis

#### **Random Forest vs XGBoost Comparison**

| Metric | Random Forest | XGBoost | Winner |
|--------|---------------|---------|--------|
| **ROC-AUC** | 0.8208 | **0.8240** | ✅ XGBoost |
| **Recall (Class 1)** | 0.13 ❌ | **0.21** | ✅ XGBoost (+62%) |
| **F1-Score (Class 1)** | 0.21 | **0.30** | ✅ XGBoost (+43%) |

**Key Decision Factors:**
- XGBoost detects **1.6x more diabetes patients** (critical in healthcare)
- Missing patients (false negatives) is more costly than false alarms in medical screening

#### **Feature Importance Discrepancy** 🔥

| Rank | Random Forest | Importance | XGBoost | Importance | Difference |
|------|---------------|------------|---------|------------|------------|
| **1st** | General Health | 0.213 | **High BP** | **0.594** 📈 | **3.1x** |
| 2nd | High BP | 0.193 | General Health | 0.124 | - |
| 3rd | BMI | 0.169 | High Cholesterol | 0.066 | - |

**Critical Finding:**
- XGBoost identified **High Blood Pressure** as the dominant predictor
- Aligns with clinical research on diabetes risk factors
- Random Forest undervalued High BP by over 3x

---

### 3.2 Risk Stratification Analysis

#### **Metabolic Syndrome Triple Threat**

| Risk Profile | Diabetes Rate | Population Size | Risk Category |
|-------------|---------------|-----------------|---------------|
| **Obese + High BP + High Cholesterol** | **44.9%** 🚨 | 25,599 | Very High |
| Overweight + High BP + High Cholesterol | 28.0% | 24,424 | High |
| Obese + High BP only | 26.7% | 16,793 | High |
| Normal BMI + All Normal | 2.7% ✅ | - | Low |

**Risk Amplification:** **16.6x difference** between highest and lowest risk groups!

#### **BMI × High Blood Pressure Interaction**

| BMI Category | Normal BP | High BP | Risk Multiplier |
|--------------|-----------|---------|-----------------|
| Normal | 3.8% | 15.5% | **4.0x** 🔴 |
| Overweight | 7.3% | 23.5% | 3.2x |
| Obese | 13.9% | 37.7% | 2.7x |

**Key Insight:** Blood pressure management is **MORE critical** for normal-weight individuals.

#### **Tiered Risk Progression**

| Stage | Risk Profile | Diabetes Rate |
|-------|-------------|---------------|
| Stage 1 | Normal + All Normal | 2.7% |
| Stage 2 | Overweight + 1 factor | 5-12% |
| Stage 3 | Obese + 1 factor | 10-27% |
| Stage 4 | Obese + 2+ factors | 28-45% 🚨 |

---

### 3.3 XGBoost Model Performance

**Final Model Metrics:**
- **ROC-AUC:** 0.8240
- **F1-Score:** 0.30
- **Recall:** 0.21 (detects more patients than Random Forest)
- **Precision:** 0.59

**Top 5 Risk Factors:**

| Rank | Feature | Importance Score | Category |
|------|---------|------------------|----------|
| 1 | **High Blood Pressure** | **0.594** 🔴 | Medical Condition |
| 2 | General Health | 0.124 | Self-Assessment |
| 3 | High Cholesterol | 0.066 | Medical Condition |
| 4 | BMI | - | Physical Metric |
| 5 | Age | - | Demographic |

---

## 4️⃣ Limitations

- **Recall still low (21%):** Model misses 79% of diabetes patients
  - Learning difficulty due to class imbalance (prediabetes 1.8%, diabetes 13.9%)
  - Healthcare applications require higher sensitivity

- **Severe class imbalance:** Prediabetes class only accounts for 1.8% of total
  - Potentially low prediction accuracy for prediabetes
  - Limited learning for minority classes

- **Survey-based data:** Self-reported health indicators may have reporting bias

- **Cross-sectional data:** Cannot capture temporal progression or causality

---

## 📊 Key Takeaways

1. **XGBoost Superior for Healthcare:** 62% better recall = more patients detected

2. **High Blood Pressure is Critical:** 3x more important than other factors

3. **Multiplicative Risk:** Three factors combined = 44.9% risk (16.6x normal)

4. **Context Matters:** Same risk factor has different impact based on other conditions

5. **Clear Action Plan:** 25,599 high-risk individuals identified for immediate intervention

---

## 📚 References

- Diabetes Health Indicators Dataset: [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv)
