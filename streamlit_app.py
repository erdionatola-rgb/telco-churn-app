# =============================================================================
# TELCO CHURN — STREAMLIT INFERENCE APP TEMPLATE
# Run with:  streamlit run streamlit_app.py
# =============================================================================
# TODO: Complete every section marked  TODO
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown("Upload a customer CSV file to get churn predictions from the trained model.")
st.markdown("---")

# =============================================================================
# 1. LOAD MODEL
# =============================================================================

@st.cache_resource
def load_pipeline():
    """Load the saved sklearn Pipeline."""
    try:
        pipeline = joblib.load("pipeline.pkl")
        return pipeline
    except FileNotFoundError:
        st.error("pipeline.pkl not found. Run notebook.ipynb first to train and save the model.")
        return None

pipeline = load_pipeline()

if pipeline is None:
    st.stop()

st.success("Model loaded successfully.")

# =============================================================================
# 2. FILE UPLOAD
# =============================================================================

st.subheader("Upload customer data")
st.markdown(
    "Upload a CSV with the same columns as the training data "
    "(**without** the `Churn` column). The `customerID` column is optional but recommended."
)

uploaded_file = st.file_uploader(
    label="Choose a CSV file",
    type=["csv"],
    help="Same schema as the Telco Customer Churn dataset, minus the Churn column.",
)

if uploaded_file is None:
    st.info("Waiting for file upload...")
    st.stop()

# =============================================================================
# 3. LOAD & VALIDATE DATA
# =============================================================================

raw_df = pd.read_csv(uploaded_file)

st.markdown(f"**Rows uploaded:** {len(raw_df):,}  |  **Columns:** {len(raw_df.columns)}")
st.dataframe(raw_df.head(5), use_container_width=True)

# Extract customerID if present
if "customerID" in raw_df.columns:
    customer_ids = raw_df["customerID"].copy()
    input_df = raw_df.drop(columns=["customerID"])
else:
    customer_ids = pd.Series(range(len(raw_df)), name="customerID")
    input_df = raw_df.copy()

# Drop Churn if accidentally included
if "Churn" in input_df.columns:
    input_df = input_df.drop(columns=["Churn"])
    st.warning("'Churn' column detected and removed — predictions are based on features only.")

# TODO: Add any preprocessing you do OUTSIDE the pipeline (e.g. TotalCharges coercion)
# These must mirror what you do in section 3 of notebook.ipynb
input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

# TODO: Add engineered features here — same logic as notebook section 3
# =============================================================================
# PREPROCESSING + FEATURE ENGINEERING
# =============================================================================

# ── Feature 1: tenure groups ──────────────────────────────────────────────
input_df["tenure_bin"] = pd.cut(
    input_df["tenure"],
    bins=[0, 12, 36, 72],
    labels=["New", "Mid", "Loyal"],
    include_lowest=True,
)

# ── Feature 2: new customer flag ──────────────────────────────────────────
input_df["is_new_customer"] = (
    input_df["tenure"] < 6
).astype(int)

# ── Feature 3: average monthly spend ──────────────────────────────────────
input_df["avg_monthly_spend"] = (
    input_df["TotalCharges"] /
    input_df["tenure"].replace(0, np.nan)
)

input_df["avg_monthly_spend"] = (
    input_df["avg_monthly_spend"]
    .fillna(input_df["MonthlyCharges"])
)

# ── Feature 4: streaming user flag ────────────────────────────────────────
input_df["is_streaming_user"] = (
    (input_df["StreamingTV"] == "Yes") |
    (input_df["StreamingMovies"] == "Yes")
).astype(int)

# ── Feature 5: high monthly charges ───────────────────────────────────────
input_df["high_monthly_charges"] = (
    input_df["MonthlyCharges"] > 80
).astype(int)

# ── Feature 6: paperless billing user ─────────────────────────────────────
input_df["paperless_user"] = (
    input_df["PaperlessBilling"] == "Yes"
).astype(int)

# ── Feature 7: many services flag ─────────────────────────────────────────
service_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

input_df["many_services"] = (
    (input_df[service_cols] == "Yes").sum(axis=1) >= 3
).astype(int)

# ── Feature 8: total services count ───────────────────────────────────────
all_service_cols = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

input_df["total_services"] = (
    (input_df[all_service_cols] == "Yes").sum(axis=1)
)

# ── Feature 9: customer value ratio ───────────────────────────────────────
input_df["customer_value_ratio"] = (
    input_df["TotalCharges"] /
    (input_df["MonthlyCharges"] + 1)
)

# ── Feature 10: charge per service ────────────────────────────────────────
input_df["charge_per_service"] = (
    input_df["MonthlyCharges"] /
    (input_df["total_services"] + 1)
)

# ── Feature 11: no support flag ───────────────────────────────────────────
input_df["no_support"] = (
    (input_df["OnlineSecurity"] == "No") &
    (input_df["TechSupport"] == "No")
).astype(int)

# ── Feature 12: fiber user ────────────────────────────────────────────────
input_df["fiber_user"] = (
    input_df["InternetService"] == "Fiber optic"
).astype(int)

# ── Feature 13: month-to-month contract ───────────────────────────────────
input_df["month_to_month"] = (
    input_df["Contract"] == "Month-to-month"
).astype(int)

# ── Feature 14: electronic payment ────────────────────────────────────────
input_df["electronic_payment"] = (
    input_df["PaymentMethod"] == "Electronic check"
).astype(int)

# ── Feature 15: senior alone ──────────────────────────────────────────────
input_df["senior_alone"] = (
    (input_df["SeniorCitizen"] == 1) &
    (input_df["Partner"] == "No") &
    (input_df["Dependents"] == "No")
).astype(int)

# ── Feature 16: high risk customer ────────────────────────────────────────
input_df["high_risk_customer"] = (
    (input_df["Contract"] == "Month-to-month") &
    (input_df["InternetService"] == "Fiber optic") &
    (input_df["PaperlessBilling"] == "Yes")
).astype(int)

# ── Feature 17: loyal high value ──────────────────────────────────────────
input_df["loyal_high_value"] = (
    (input_df["tenure"] > 24) &
    (input_df["MonthlyCharges"] > 70)
).astype(int)

# ── Feature 18: low engagement ────────────────────────────────────────────
input_df["low_engagement"] = (
    input_df["total_services"] <= 2
).astype(int)

# ── Feature 19: auto payment ──────────────────────────────────────────────
input_df["auto_payment"] = (
    input_df["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
).astype(int)

# ── Feature 20: expensive fiber ───────────────────────────────────────────
input_df["expensive_fiber"] = (
    (input_df["InternetService"] == "Fiber optic") &
    (input_df["MonthlyCharges"] > 90)
).astype(int)

# =============================================================================
# 4. RUN PREDICTIONS
# =============================================================================

if st.button("Run predictions", type="primary"):

    with st.spinner("Running model inference..."):
        try:
            preds = pipeline.predict(input_df)
            proba = pipeline.predict_proba(input_df)[:, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Build results dataframe
    results = pd.DataFrame({
        "customerID":        customer_ids.values,
        "Churn_predicted":   preds,
        "Churn_label":       ["Churn" if p == 1 else "No Churn" for p in preds],
        "Churn_probability": proba.round(4),
    })

    # ── 4a. Summary metrics ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total customers", f"{len(results):,}")
    col2.metric("Predicted to churn",  f"{preds.sum():,}")
    col3.metric("Predicted to stay",   f"{(preds == 0).sum():,}")
    col4.metric("Predicted churn rate", f"{preds.mean():.1%}")

    # ── 4b. Predictions table ──────────────────────────────────────────────
    st.subheader("Predictions")

    # Colour code the table
    def highlight_churn(row):
        if row["Churn_predicted"] == 1:
            return ["background-color: #FEF2F2"] * len(row)
        return ["background-color: #F0FDF4"] * len(row)

    styled = results.style.apply(highlight_churn, axis=1) \
        .format({"Churn_probability": "{:.4f}"})

    st.dataframe(styled, use_container_width=True)

    # ── 4c. Churn probability distribution ───────────────────────────────
    st.subheader("Churn probability distribution")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(proba, bins=30, color="#0D9488", edgecolor="white", alpha=0.85)
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Decision boundary (0.5)")
    axes[0].set_xlabel("Churn probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of churn probabilities")
    axes[0].legend()

    # Pie chart
    churn_counts = results["Churn_label"].value_counts()
    axes[1].pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=["#0D9488", "#EF4444"],
        startangle=90,
    )
    axes[1].set_title("Predicted churn split")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # TODO: Add more visualizations here
    # Ideas: high-risk customer table (prob > 0.7), risk segments bar chart

    # ── 4d. Download predictions ──────────────────────────────────────────
    st.subheader("Download results")

    download_df = results[["customerID", "Churn_predicted", "Churn_probability"]]
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download predictions.csv",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.success("Done! Download the CSV above to submit for the competition.")

# High-risk customers
st.subheader("High-risk customers")

high_risk = results[results["Churn_probability"] > 0.7]

st.dataframe(
    high_risk.sort_values(
        by="Churn_probability",
        ascending=False
    ),
    use_container_width=True
)

# Risk segments
st.subheader("Risk segments")

risk_bins = pd.cut(
    results["Churn_probability"],
    bins=[0, 0.3, 0.7, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

risk_counts = risk_bins.value_counts().sort_index()

fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.bar(
    risk_counts.index.astype(str),
    risk_counts.values
)

ax2.set_xlabel("Risk Segment")
ax2.set_ylabel("Customers")
ax2.set_title("Customer Risk Segments")

st.pyplot(fig2)
plt.close()