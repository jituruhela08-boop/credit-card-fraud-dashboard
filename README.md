# 🛡️ MuleShield AI — Credit Card Fraud Detection Dashboard

> **Task 10 · Data Science Dashboard**  
> IK Gujral Punjab Technical University

---

## 📌 Project Overview

An end-to-end **Data Science Dashboard** built with **Streamlit** that trains, evaluates, and visualizes three Machine Learning models for Credit Card Fraud Detection on the Kaggle `creditcard.csv` dataset.

The dashboard is fully interactive, professionally styled, and ready for college submission or portfolio presentation.

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd credit-card-fraud-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place `creditcard.csv` in the **same folder** as `app.py`.

> ⚠️ If `creditcard.csv` is not found, the dashboard automatically generates a **synthetic dataset** with identical structure so you can still run and demo all features.

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📂 Project Structure

```
credit-card-fraud-dashboard/
│
├── app.py               ← Main Streamlit dashboard
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── creditcard.csv       ← Dataset (add manually)
```

---

## 🗂️ Dashboard Pages

| Page | Description |
|---|---|
| **📊 Dataset Overview** | KPI cards, data table, statistical summary, class distribution donut |
| **📈 Visualizations** | Bar chart, pie chart, histogram, correlation heatmap, box plot |
| **🤖 ML Model Results** | Accuracy badges, grouped bar comparison, radar chart |
| **📋 Evaluation Metrics** | Confusion matrix, per-metric breakdown, all-model summary table |
| **💡 Insights & Recommendations** | Business findings and actionable production recommendations |

---

## 🤖 Machine Learning Models

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline linear classifier, fast & interpretable |
| **Decision Tree** | Non-linear, max depth = 8, prone to overfitting |
| **Random Forest** | Ensemble of 100 trees, typically best performer |

All models use:
- 80/20 stratified train-test split
- `StandardScaler` feature normalization
- Evaluated on: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## 📊 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total Records | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Features | 30 (V1–V28, Time, Amount) |
| Class Imbalance | ~578:1 (genuine:fraud) |

Features V1–V28 are PCA-transformed for anonymization. Only `Time` and `Amount` are raw.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `matplotlib` | Static plotting support |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML models & metrics |

---

## 💡 Key Findings

- Only **0.17%** of transactions are fraudulent — severe class imbalance
- **Random Forest** achieves the best F1 Score and generalization
- Fraudulent transactions cluster at **lower amounts** (< $200)
- Features **V4, V11, V12, V14** correlate most strongly with fraud
- **False Negatives** (missed fraud) are far costlier than False Positives

---

## 🏫 Submission Details

- **University:** IK Gujral Punjab Technical University  
- **Task:** Task 10 — Data Science Dashboard  
- **Project:** MuleShield AI — Financial Crime Detection  

---

## 📄 License

This project is submitted for academic purposes.  
Dataset © Worldline and Université Libre de Bruxelles (ULB) — used under Kaggle terms.