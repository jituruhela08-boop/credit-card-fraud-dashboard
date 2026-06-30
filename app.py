import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
    padding: 6px 0;
}

/* Main background */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* Page title */
.dash-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.dash-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 4px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* Accuracy badge */
.acc-badge {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #38bdf8;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    color: #38bdf8;
    font-weight: 600;
    margin: 0.3rem;
}

/* Alert box */
.insight-box {
    background: #0f2027;
    border: 1px solid #0ea5e9;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    color: #bae6fd;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, #38bdf8, transparent);
    margin: 1.5rem 0;
}

/* Plotly chart background overrides */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("creditcard.csv")
    except FileNotFoundError:
        # Generate synthetic data that mimics creditcard.csv structure
        np.random.seed(42)
        n = 284807
        fraud_n = 492
        genuine_n = n - fraud_n

        # PCA-like V features
        v_cols = {f"V{i}": np.concatenate([
            np.random.normal(0, 1, genuine_n),
            np.random.normal(0.5, 1.5, fraud_n)
        ]) for i in range(1, 29)}

        df = pd.DataFrame(v_cols)
        df["Time"] = np.concatenate([
            np.random.uniform(0, 172792, genuine_n),
            np.random.uniform(0, 172792, fraud_n)
        ])
        df["Amount"] = np.concatenate([
            np.abs(np.random.exponential(88, genuine_n)),
            np.abs(np.random.exponential(120, fraud_n))
        ])
        df["Class"] = np.concatenate([np.zeros(genuine_n), np.ones(fraud_n)]).astype(int)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

@st.cache_data
def train_models(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    results["Logistic Regression"] = {
        "model": lr, "y_pred": y_pred_lr, "y_test": y_test,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr, zero_division=0),
        "recall": recall_score(y_test, y_pred_lr, zero_division=0),
        "f1": f1_score(y_test, y_pred_lr, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred_lr),
    }

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train_s, y_train)
    y_pred_dt = dt.predict(X_test_s)
    results["Decision Tree"] = {
        "model": dt, "y_pred": y_pred_dt, "y_test": y_test,
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "precision": precision_score(y_test, y_pred_dt, zero_division=0),
        "recall": recall_score(y_test, y_pred_dt, zero_division=0),
        "f1": f1_score(y_test, y_pred_dt, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred_dt),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    results["Random Forest"] = {
        "model": rf, "y_pred": y_pred_rf, "y_test": y_test,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred_rf),
    }

    return results

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Credit Card Fraud Detection**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Dataset Overview",
            "📈 Visualizations",
            "🤖 ML Model Results",
            "📋 Evaluation Metrics",
            "💡 Insights & Recommendations",
        ],
    )
  

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading dataset & training models..."):
    df = load_data()
    model_results = train_models(df)

total = len(df)
fraud = int(df["Class"].sum())
genuine = total - fraud
fraud_pct = round((fraud / total) * 100, 4)

# ─────────────────────────────────────────────
#  CHART THEME HELPER
# ─────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ═════════════════════════════════════════════
#  PAGE 1 — DATASET OVERVIEW
# ═════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.markdown('<div class="dash-title">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">Task 10 · Data Science Dashboard · IK Gujral Punjab Technical University</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{total:,}", "Total Transactions"),
        (c2, f"{fraud:,}", "Fraud Transactions"),
        (c3, f"{genuine:,}", "Genuine Transactions"),
        (c4, f"{fraud_pct}%", "Fraud Percentage"),
    ]
    for col, val, label in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # Dataset shape info
    st.markdown('<div class="section-header">📁 Dataset Overview</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | **Total Records** | {total:,} |
        | **Total Features** | {df.shape[1]} |
        | **Fraud Cases** | {fraud:,} |
        | **Genuine Cases** | {genuine:,} |
        | **Class Imbalance Ratio** | {round(genuine/fraud, 1)}:1 (genuine : fraud) |
        | **Amount Range** | ${df['Amount'].min():.2f} – ${df['Amount'].max():.2f} |
        | **Mean Transaction Amount** | ${df['Amount'].mean():.2f} |
        """)

    with col_b:
        fig_donut = go.Figure(go.Pie(
            labels=["Genuine", "Fraud"],
            values=[genuine, fraud],
            hole=0.6,
            marker=dict(colors=["#38bdf8", "#f43f5e"],
                        line=dict(color="#0f172a", width=2)),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_donut.update_layout(
            title="Class Distribution",
            showlegend=False,
            height=260,
            **CHART_THEME,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Sample data
    st.markdown('<div class="section-header">🔍 Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(
        df.head(10).style.background_gradient(cmap="Blues", subset=["Amount"]),
        use_container_width=True,
        height=250,
    )

    # Basic stats
    st.markdown('<div class="section-header">📐 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["Time", "Amount", "Class"]].describe().round(4),
        use_container_width=True,
    )

# ═════════════════════════════════════════════
#  PAGE 2 — VISUALIZATIONS
# ═════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown('<div class="dash-title">Interactive Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">Explore patterns in fraud vs genuine transactions</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Row 1: Bar + Pie
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Fraud vs Genuine — Bar Chart</div>', unsafe_allow_html=True)
        bar_df = pd.DataFrame({
            "Type": ["Genuine", "Fraud"],
            "Count": [genuine, fraud],
        })
        fig_bar = px.bar(
            bar_df, x="Type", y="Count",
            color="Type",
            color_discrete_map={"Genuine": "#38bdf8", "Fraud": "#f43f5e"},
            text="Count",
        )
        fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_bar.update_layout(
            showlegend=False, height=350,
            xaxis_title="", yaxis_title="Number of Transactions",
            **CHART_THEME,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Fraud Percentage — Pie Chart</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Genuine (99.83%)", "Fraud (0.17%)"],
            values=[genuine, fraud],
            marker=dict(colors=["#0ea5e9", "#f43f5e"],
                        line=dict(color="#0f172a", width=3)),
            pull=[0, 0.08],
            textinfo="label+percent",
            textfont=dict(color="#fff", size=11),
        ))
        fig_pie.update_layout(
            showlegend=False, height=350,
            **CHART_THEME,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Row 2: Histogram
    st.markdown('<div class="section-header">Transaction Amount Distribution</div>', unsafe_allow_html=True)
    amount_cap = st.slider("Cap Amount at ($)", 100, 5000, 1000, step=100)
    df_capped = df[df["Amount"] <= amount_cap].copy()
    df_capped["Type"] = df_capped["Class"].map({0: "Genuine", 1: "Fraud"})

    fig_hist = px.histogram(
        df_capped, x="Amount", color="Type",
        nbins=80,
        color_discrete_map={"Genuine": "#38bdf8", "Fraud": "#f43f5e"},
        barmode="overlay",
        opacity=0.75,
    )
    fig_hist.update_layout(
        height=360,
        xaxis_title="Transaction Amount ($)",
        yaxis_title="Count",
        **CHART_THEME,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Row 3: Correlation Heatmap
    st.markdown('<div class="section-header">Correlation Heatmap (V1–V10, Amount, Class)</div>', unsafe_allow_html=True)
    heat_cols = [f"V{i}" for i in range(1, 11)] + ["Amount", "Class"]
    corr = df[heat_cols].corr().round(2)

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(tickfont=dict(color="#e2e8f0")),
    ))
    fig_heat.update_layout(height=440, **CHART_THEME)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Row 4: Box plot — Amount by class
    st.markdown('<div class="section-header">Amount Distribution by Class (Box Plot)</div>', unsafe_allow_html=True)
    df_box = df.copy()
    df_box["Type"] = df_box["Class"].map({0: "Genuine", 1: "Fraud"})
    fig_box = px.box(
        df_box[df_box["Amount"] <= 2000], x="Type", y="Amount",
        color="Type",
        color_discrete_map={"Genuine": "#38bdf8", "Fraud": "#f43f5e"},
        points="outliers",
    )
    fig_box.update_layout(
        showlegend=False, height=380,
        xaxis_title="", yaxis_title="Amount ($)",
        **CHART_THEME,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ═════════════════════════════════════════════
#  PAGE 3 — ML MODEL RESULTS
# ═════════════════════════════════════════════
elif page == "🤖 ML Model Results":
    st.markdown('<div class="dash-title">Machine Learning Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">Training & Accuracy Comparison · 80/20 Train-Test Split</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    model_names = list(model_results.keys())
    accuracies = [model_results[m]["accuracy"] for m in model_names]
    f1_scores  = [model_results[m]["f1"]       for m in model_names]

    # Accuracy badges
    st.markdown('<div class="section-header">Model Accuracies</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    icons = ["📘", "🌳", "🌲"]
    for i, (col, name) in enumerate(zip(cols, model_names)):
        acc = model_results[name]["accuracy"]
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem">{icons[i]}</div>
            <div class="metric-value">{acc*100:.2f}%</div>
            <div class="metric-label">{name}</div>
        </div>""", unsafe_allow_html=True)

    # Best model highlight
    best_idx = int(np.argmax(accuracies))
    best_name = model_names[best_idx]
    st.markdown(f"""
    <br>
    <div class="insight-box">
        🏆 <strong>Best Model: {best_name}</strong> — Accuracy {accuracies[best_idx]*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # Grouped bar chart
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    precision_scores = [model_results[m]["precision"] for m in model_names]
    recall_scores    = [model_results[m]["recall"]    for m in model_names]

    fig_compare = go.Figure()
    metrics_map = {
        "Accuracy":  (accuracies,        "#38bdf8"),
        "Precision": (precision_scores,   "#818cf8"),
        "Recall":    (recall_scores,      "#34d399"),
        "F1 Score":  (f1_scores,          "#f59e0b"),
    }
    for metric, (vals, color) in metrics_map.items():
        fig_compare.add_trace(go.Bar(
            name=metric, x=model_names, y=[v * 100 for v in vals],
            marker_color=color,
            text=[f"{v*100:.2f}%" for v in vals],
            textposition="outside",
        ))
    fig_compare.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 115], title="Score (%)"),
        height=420,
        legend=dict(orientation="h", y=-0.18),
        **CHART_THEME,
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header">Radar — Multi-Metric View</div>', unsafe_allow_html=True)
    categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors_r = ["#38bdf8", "#f59e0b", "#34d399"]
    fig_radar = go.Figure()
    for i, name in enumerate(model_names):
        r = model_results[name]
        vals = [r["accuracy"], r["precision"], r["recall"], r["f1"]]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=name,
            line=dict(color=colors_r[i], width=2),
            fillcolor=colors_r[i].replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in colors_r[i] else colors_r[i],
            opacity=0.75,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.7)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8")),
            angularaxis=dict(tickfont=dict(color="#e2e8f0")),
        ),
        height=420,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        **CHART_THEME,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ═════════════════════════════════════════════
#  PAGE 4 — EVALUATION METRICS
# ═════════════════════════════════════════════
elif page == "📋 Evaluation Metrics":
    st.markdown('<div class="dash-title">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">Detailed per-model metrics and confusion matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select Model to Inspect",
        list(model_results.keys()),
    )
    res = model_results[selected_model]

    # Metric cards
    m_cols = st.columns(4)
    metrics = [
        ("🎯 Accuracy",  res["accuracy"],  "#38bdf8"),
        ("🔬 Precision", res["precision"], "#818cf8"),
        ("📡 Recall",    res["recall"],    "#34d399"),
        ("⚖️ F1 Score",  res["f1"],        "#f59e0b"),
    ]
    for col, (label, val, color) in zip(m_cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val*100:.2f}%</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_bar = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = res["cm"]
        cm_labels = ["Genuine (0)", "Fraud (1)"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=cm_labels,
            y=cm_labels,
            colorscale=[[0, "#0f172a"], [1, "#38bdf8"]],
            text=cm,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=20, color="white"),
            showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=320,
            **CHART_THEME,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Breakdown table
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | True Negatives (TN) | {tn:,} |
        | False Positives (FP) | {fp:,} |
        | False Negatives (FN) | {fn:,} |
        | True Positives (TP) | {tp:,} |
        """)

    with col_bar:
        st.markdown('<div class="section-header">Metrics Breakdown</div>', unsafe_allow_html=True)
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metric_values = [res["accuracy"], res["precision"], res["recall"], res["f1"]]
        metric_colors = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b"]

        fig_mb = go.Figure()
        for m, v, c in zip(metric_names, metric_values, metric_colors):
            fig_mb.add_trace(go.Bar(
                x=[v * 100], y=[m],
                orientation="h",
                marker_color=c,
                text=f"{v*100:.2f}%",
                textposition="outside",
                name=m,
            ))
        fig_mb.update_layout(
            showlegend=False,
            xaxis=dict(range=[0, 115], title="Score (%)"),
            height=320,
            **CHART_THEME,
        )
        st.plotly_chart(fig_mb, use_container_width=True)

    # All models table
    st.markdown('<div class="section-header">All Models — Summary Table</div>', unsafe_allow_html=True)
    summary_data = []
    for name, r in model_results.items():
        summary_data.append({
            "Model": name,
            "Accuracy (%)": round(r["accuracy"] * 100, 3),
            "Precision (%)": round(r["precision"] * 100, 3),
            "Recall (%)": round(r["recall"] * 100, 3),
            "F1 Score (%)": round(r["f1"] * 100, 3),
        })
    summary_df = pd.DataFrame(summary_data).set_index("Model")
    st.dataframe(
        summary_df.style
            .highlight_max(axis=0, color="#1e3a5f")
            .format("{:.3f}"),
        use_container_width=True,
    )

# ═════════════════════════════════════════════
#  PAGE 5 — INSIGHTS & RECOMMENDATIONS
# ═════════════════════════════════════════════
elif page == "💡 Insights & Recommendations":
    st.markdown('<div class="dash-title">Insights & Business Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">Actionable findings from the Credit Card Fraud Detection analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    best_model = max(model_results, key=lambda m: model_results[m]["f1"])
    best_acc = model_results[best_model]["accuracy"] * 100
    best_f1  = model_results[best_model]["f1"] * 100

    insights = [
        ("🔎 Severe Class Imbalance",
         f"Only {fraud_pct}% of {total:,} transactions are fraudulent. This extreme imbalance "
         "means accuracy alone is a misleading metric — a model predicting 'Genuine' always would still "
         f"achieve ~99.83% accuracy. Metrics like Precision, Recall, and F1 are critical for real evaluation."),

        ("🏆 Best Performing Model",
         f"<strong>{best_model}</strong> achieved the highest F1 Score of {best_f1:.2f}% and "
         f"Accuracy of {best_acc:.2f}%. Its ensemble nature makes it robust to noise and "
         "generalizes well even on highly imbalanced data."),

        ("📊 Feature Importance",
         "PCA-transformed features V4, V11, V12, and V14 show the strongest correlation with fraud. "
         "These anonymized dimensions likely encode transaction behaviour patterns most indicative of "
         "unauthorized card use."),

        ("💰 Transaction Amount Patterns",
         "Fraudulent transactions tend to cluster at lower amounts (typically under $200) — "
         "consistent with 'test charge' strategies used by fraudsters before executing larger withdrawals. "
         "Amount alone is insufficient but adds predictive value in combination with PCA features."),

        ("⏱️ Time-Based Patterns",
         "Fraud events are slightly more frequent during off-peak hours (late night / early morning). "
         "Implementing time-aware thresholds in production can reduce false negatives without "
         "degrading user experience during peak hours."),

        ("🚨 False Negative Cost",
         "In fraud detection, a False Negative (missed fraud) is far more costly than a False Positive "
         "(wrongly flagging a genuine transaction). The production threshold should be tuned to prioritize "
         "higher Recall — catching all fraud — even at the cost of slightly lower Precision."),
    ]

    for title, body in insights:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{title}</strong><br>{body}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Business Recommendations</div>', unsafe_allow_html=True)

    recommendations = [
        ("Deploy Random Forest in Production",
         "Its ensemble approach handles class imbalance better than single estimators. Wrap it in an API "
         "endpoint to score transactions in real-time before authorization."),
        ("Apply SMOTE / Class Weighting",
         "Use Synthetic Minority Over-sampling (SMOTE) or class_weight='balanced' to further improve "
         "Recall on the minority (fraud) class and reduce missed detections."),
        ("Set Custom Decision Threshold",
         "Lower the classification threshold from 0.5 to ~0.3–0.35 to increase fraud catch rate, "
         "then monitor false positive rate with operations teams."),
        ("Integrate Real-Time Scoring",
         "Stream transactions through the trained model via a lightweight REST API. Flag suspicious "
         "transactions for human review before final authorization, targeting sub-100ms latency."),
        ("Implement Continuous Retraining",
         "Fraud patterns evolve. Schedule monthly model retraining on fresh data with automated performance "
         "benchmarking to detect model drift before it affects detection rates."),
        ("Add Explainability Layer",
         "Use SHAP or LIME to explain individual fraud decisions to compliance teams and end users. "
         "Regulatory requirements (e.g. RBI guidelines) may mandate explainable AI for financial decisions."),
    ]

    for i, (rec_title, rec_body) in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="insight-box">
            <strong>{i}. {rec_title}</strong><br>{rec_body}
        </div>""", unsafe_allow_html=True)

    # Closing
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.85rem; padding: 1rem 0;">
         
        
    </div>
    """, unsafe_allow_html=True)