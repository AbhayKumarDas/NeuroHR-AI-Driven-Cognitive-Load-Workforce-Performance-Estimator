# NeuroHR: AI-Driven Cognitive Load & Workforce Performance Estimator

An ML-Based Framework for Burnout Prediction, Performance Impact Estimation, and HR Decision Support.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

**NeuroHR** is a Machine Learning framework that predicts cognitive load, burnout probability, and performance drop using engineered HR behavioral signals, ensemble regression models, and unsupervised clustering to generate data-driven task allocation insights for workforce optimization.

This tool estimates the **Cognitive Load Index (CLI)** of employees using behavioral and workload indicators, helping HR identify burnout risks, predict performance impact, and make data-driven decisions on task assignments.

## Key Features

- **CLI Prediction**: Random Forest Regressor with 97% R² accuracy
- **Performance Impact Estimation**: Predicts performance drop when new tasks are assigned
- **Employee Clustering**: K-Means clustering for behavioral segmentation
- **HR Decision Support**: Automated recommendations for task allocation
- **Interactive Web App**: Streamlit-based tool for real-time analysis

---

## Project Structure

```
NeuroHR/
├── notebook/
│   └── NeuroHR.ipynb          # Main ML notebook (Core Implementation)
├── models/
│   ├── cli_model.pkl          # Trained CLI regression model
│   ├── delta_model.pkl        # Performance drop estimator
│   └── clustering_artifacts.pkl # K-Means clustering artifacts
├── data/
│   └── hr_cognitive_load_dataset.csv  # Dataset (10K employees)
├── reports/
│   ├── figures/               # Visualization outputs
│   └── NeuroHR_Presentation.pdf
|   └── Generated_Sample_Report.pdf
|
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Machine Learning Pipeline

### 1. Dataset Generation (10,000 Synthetic Employees)

The notebook generates a comprehensive HR dataset with **61 features** including:

| Category | Features |
|----------|----------|
| **Demographics** | Employee_ID, Job_Title, Department, Role_Level, Employee_Persona |
| **Performance** | Performance_Rating_Current, Perf_Trend_Slope, Performance_Volatility |
| **Workload** | Weekly_Workload_Hours, Active_Task_Count, Overtime_Hours_30d |
| **Behavioral** | Self_Reported_Stress, Sentiment_Score, Focus_Time_Ratio |
| **Target Variables** | CLI_v2, Risk_Category, Delta_if_Assign, Recommendation |

### 2. Feature Engineering

**Cognitive Load Index (CLI) Formula:**
```
CLI = 0.20 × Workload_Ratio + 0.15 × Stress_Norm + 0.12 × Focus_Inv +
      0.10 × Task_Switch + 0.10 × Sentiment_Inv + 0.08 × Overtime + ...
```

**Employee Personas** with capacity multipliers:
- Resilient (1.25x), Balanced (1.0x), Workaholic (1.1x), Sensitive (0.85x), Struggling (0.7x)

### 3. Model Training

#### CLI Regression Model (Random Forest)
```
Features: 24 numerical + 4 categorical
Architecture: RandomForestRegressor(n_estimators=200, max_depth=15)
Preprocessing: ColumnTransformer + OneHotEncoder + Pipeline
```

| Metric | Training | Testing |
|--------|----------|---------|
| R² | 0.9965 | 0.9717 |
| MAE | 0.0051 | 0.0143 |
| RMSE | 0.0067 | 0.0191 |

#### Performance Drop Estimator (Random Forest)
```
Features: 17 numerical + 4 categorical (includes CLI_v2)
Target: Delta_if_Assign (expected performance change)
```

| Metric | Training | Testing |
|--------|----------|---------|
| R² | 0.9967 | 0.9595 |
| MAE | 0.0157 | 0.0443 |
| RMSE | 0.0203 | 0.0580 |

#### Employee Clustering (K-Means)
```
Features: 11 behavioral indicators
Algorithm: KMeans(n_clusters=4) + StandardScaler + PCA
```

**Cluster Interpretations:**
- Cluster 0: Healthy & Balanced
- Cluster 1: Stable & Low Risk
- Cluster 2: Moderate Risk
- Cluster 3: High Workload & Stress

---

## Cognitive Load Index (CLI)

CLI is a quantitative measure (0–1) of an employee's mental and workload stress level:

| CLI Range | Risk Level | Description |
|-----------|------------|-------------|
| 0.00 - 0.30 | Low | Employee is balanced and can handle more responsibility |
| 0.30 - 0.55 | Medium | Approaching moderate workload stress. Monitor closely |
| 0.55 - 0.75 | High | Signs of fatigue and performance decline. Support needed |
| 0.75 - 1.00 | Critical | Burnout risk. Immediate intervention required |

**Burnout Probability Formula:**
```python
burnout_prob = 1 / (1 + exp(-10 × (CLI - 0.55)))
```

---

## HR Recommendation Engine

The system provides actionable recommendations based on CLI, Delta, and Capacity scores:

| Recommendation | Condition |
|----------------|-----------|
| **Do Not Assign** | CLI ≥ 0.75 OR Delta ≤ -1.0 OR Capacity < 0.7 |
| **Assign With Support** | CLI ≥ 0.55 OR Delta ≤ -0.5 OR Skill_Match < 0.5 |
| **High Priority** | Skill_Match > 0.75 AND CLI < 0.40 AND Delta > -0.3 |
| **Assign Normally** | Default case |

---

## Tech Stack

**Python, Streamlit, Scikit-learn (Random Forest Regressor, KMeans Clustering, PCA, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, Cross-Validation), Pandas, NumPy, Matplotlib, Seaborn, FPDF (PDF Report Generation), Pickle (Model Serialization)**

---

## Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/NeuroHR.git
cd NeuroHR
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook** (for ML exploration)
```bash
jupyter notebook notebook/NeuroHR.ipynb
```

4. **Run the Streamlit App** (for HR tool)
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Notebook Walkthrough

The `notebook/NeuroHR.ipynb` contains the complete ML pipeline:

| Cell | Description |
|------|-------------|
| **Cell 1** | Dataset Generation - Creates 10K employees with 61 features |
| **Cell 2** | Data Exploration - Overview, sample records, validation |
| **Cell 3** | Column Analysis - Feature inspection |
| **Cell 4** | CLI Regression Model - Training, evaluation, cross-validation |
| **Cell 5** | Performance Drop Estimator - Delta prediction model |
| **Cell 6** | Employee Clustering - K-Means with PCA visualization |
| **Cell 7** | Multi-Scenario Simulation - 4 employee case studies |
| **Cell 8** | Interactive Prediction - Real-time input and visualization |

---

## Model Performance Summary

| Model | Task | Test R² | Status |
|-------|------|---------|--------|
| CLI Regressor | Cognitive Load Prediction | 97.17% | Good Generalization |
| Delta Estimator | Performance Impact Prediction | 95.95% | Good Generalization |
| K-Means Clustering | Employee Segmentation | - | 4 Distinct Clusters |

**Cross-Validation Results (5-Fold):**
- CLI Model: Mean R² = 0.9623 (±0.0041)
- Delta Model: Mean R² = 0.9580 (±0.0035)

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Employees | 10,000 |
| Total Features | 61 |
| CLI Range | 0.074 - 0.742 |
| Average Stress Level | 3.83/5.0 |
| Average Workload | 40.2 hrs/week |
| Attrition Rate | 19.85% |

**Risk Distribution:**
- Low: 39.9%
- Medium: 55.3%
- High: 4.8%

**Recommendation Distribution:**
- Assign With Support: 41.9%
- Do Not Assign: 35.1%
- Assign Normally: 18.3%
- High Priority: 4.8%

---

## Sample Simulation Results

| Employee | Role | CLI | Burnout | Delta | Risk | Action |
|----------|------|-----|---------|-------|------|--------|
| Aditi Rao | Mid | 0.287 | 6.4% | -0.540 | Low | Assign With Support |
| Rohan Mehta | Senior | 0.458 | 28.5% | -0.837 | Medium | Assign With Support |
| Sara Ahmed | Lead | 0.621 | 66.2% | -0.948 | High | Do Not Assign |
| Arjun Patel | C-Level | 0.842 | 94.7% | -1.124 | Critical | Do Not Assign |

---

## Use Cases

1. **Workload Balancing** - Identify overloaded employees and redistribute tasks
2. **Burnout Prevention** - Early detection of burnout risks
3. **Task Assignment** - Optimize task allocation based on employee capacity
4. **Retention Strategy** - Predict and prevent employee attrition
5. **HR Analytics** - Data-driven workforce planning and wellbeing monitoring

---

## Future Enhancements

- Real-time integration with HR systems (SAP, Workday)
- Deep learning models for improved accuracy
- Sentiment analysis from communication tools (Slack, Email)
- Wearable device integration for physiological metrics
- Personalized intervention recommendations

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

Built with Scikit-learn, Streamlit, and Python data science ecosystem.

---

**Note:** This is a simulation-based project using synthetic data. For production deployment, integrate with real HR data sources and comply with privacy regulations (GDPR, data anonymization, etc.).
