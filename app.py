import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from lifetimes import BetaGeoFitter, GammaGammaFitter
from scipy.stats import ttest_ind

st.set_page_config(page_title="Nykaa Analytics Dashboard", layout="wide")

# ----------- Load Data ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("NYKA.csv")
    return data

data = load_data()

st.title("Nykaa Customer Analytics Dashboard")
st.markdown("""
This dashboard helps Nykaa understand customer segmentation, predict CLTV, 
analyze churn, and simulate A/B testing outcomes. 
""")

# ----------- Sidebar Navigation ---------------
section = st.sidebar.radio(
    "Select Analysis Section:",
    ("Data Overview", "Customer Segmentation", "CLTV Prediction", "Churn Prediction", "A/B Testing Simulation")
)

# ================================================
# SECTION 1: Data Overview
# ================================================
if section == "Data Overview":
    st.header("Data Overview")
    st.markdown("We first explore the raw data to understand its quality and trends.")

    st.subheader("Raw Data Sample")
    st.dataframe(data.head())

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Missing Values Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

# ================================================
# SECTION 2: Customer Segmentation
# ================================================
elif section == "Customer Segmentation":
    st.header("Customer Segmentation using RFM + KMeans")
    st.markdown("""
    Segment customers using Recency, Frequency, Monetary (RFM) analysis 
    and KMeans clustering.
    """)

    # Example columns - adjust to your CSV
    customer_col = 'CustomerID'
    date_col = 'InvoiceDate'
    amount_col = 'TotalAmount'

    data[date_col] = pd.to_datetime(data[date_col])
    snapshot_date = data[date_col].max() + pd.Timedelta(days=1)

    rfm = data.groupby(customer_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_col: 'count',
        amount_col: 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    st.subheader("RFM Table")
    st.dataframe(rfm.head())

    # KMeans Clustering
    k = st.slider("Select number of clusters:", 2, 8, 4)
    model = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = model.fit_predict(rfm)

    st.subheader("Cluster Counts")
    st.bar_chart(rfm['Cluster'].value_counts())

    st.subheader("Pairplot of Clusters")
    fig = sns.pairplot(rfm.reset_index(), hue='Cluster')
    st.pyplot(fig)

# ================================================
# SECTION 3: CLTV Prediction
# ================================================
elif section == "CLTV Prediction":
    st.header("Customer Lifetime Value (CLTV) Prediction")
    st.markdown("""
    Forecast future revenue using simplified BG/NBD and Gamma-Gamma models.
    (Requires frequency, recency, T, monetary value data.)
    """)

    st.warning("For demo purposes, this section uses synthetic summary features.")

    # Generate demo RFM summary
    np.random.seed(42)
    summary = pd.DataFrame({
        'frequency': np.random.randint(1, 10, 200),
        'recency': np.random.randint(1, 100, 200),
        'T': np.random.randint(90, 200, 200),
        'monetary_value': np.random.uniform(10, 1000, 200)
    })

    # Fit BG/NBD
    bgf = BetaGeoFitter()
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

    ggf = GammaGammaFitter()
    ggf.fit(summary['frequency'], summary['monetary_value'])

    summary['CLTV'] = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=3,
        freq='W',
        discount_rate=0.01
    )

    st.subheader("CLTV Distribution")
    fig, ax = plt.subplots()
    summary['CLTV'].hist(bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Predicted CLTV Customers")
    st.dataframe(summary.sort_values('CLTV', ascending=False).head(10))

# ================================================
# SECTION 4: Churn Prediction
# ================================================
elif section == "Churn Prediction":
    st.header("Churn Prediction for First-Time Buyers")
    st.markdown("""
    Train a basic churn classifier to predict which customers are likely 
    to churn after their first purchase.
    (Here we simulate the dataset for demo.)
    """)

    st.warning("Replace this simulation with real churn-ready features in practice!")

    # Generate synthetic churn dataset
    np.random.seed(42)
    N = 500
    X = pd.DataFrame({
        'AvgCartValue': np.random.uniform(100, 1000, N),
        'SessionLength': np.random.uniform(1, 30, N),
        'PagesViewed': np.random.randint(1, 20, N),
        'DiscountUsed': np.random.randint(0, 2, N)
    })
    y = np.random.randint(0, 2, N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    # Feature Importance
    st.subheader("Feature Importances")
    fig, ax = plt.subplots()
    ax.barh(X_train.columns, model.feature_importances_)
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.legend()
    st.pyplot(fig)

# ================================================
# SECTION 5: A/B Testing Simulation
# ================================================
elif section == "A/B Testing Simulation":
    st.header("A/B Testing Simulation")
    st.markdown("""
    Simulate an A/B test to compare control vs. treatment groups 
    in retention rates.
    """)

    np.random.seed(42)
    N = 500
    control = np.random.normal(0.20, 0.05, N)
    treatment = np.random.normal(0.25, 0.05, N)

    st.subheader("Retention Rate Distribution")
    fig, ax = plt.subplots()
    ax.hist(control, alpha=0.6, label='Control', bins=20)
    ax.hist(treatment, alpha=0.6, label='Treatment', bins=20)
    ax.legend()
    st.pyplot(fig)

    t_stat, p_val = ttest_ind(treatment, control)
    st.subheader("T-Test Result")
    st.write(f"T-statistic: {t_stat:.2f}")
    st.write(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        st.success("Statistically significant uplift detected!")
    else:
        st.warning("No significant difference detected.")
