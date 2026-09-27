import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Cybersecurity SOC",
    page_icon="🛡️",
    layout="wide"
)
# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #777;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f7fa;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ AI Cybersecurity SOC</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Security Operations Center for Threat Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# LOAD DATA
# =========================================================
try:
    df = pd.read_csv("data/alerts.csv")
except FileNotFoundError:
    st.error("❌ alerts.csv not found inside the data folder.")
    st.stop()
# =========================================================
# PHASE 7 - REAL-TIME MONITORING
# =========================================================

def refresh_dashboard():
    st.session_state.last_refresh = datetime.now()
    st.session_state.refresh_count += 1


if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0


st.button(
    "🔄 Refresh Security Data",
    on_click=refresh_dashboard
)

st.caption(
    f"🕒 Last refreshed: "
    f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
)

st.caption(
    f"🔄 Refresh count: {st.session_state.refresh_count}"
)
    # ---------------------------------------------------------
# LIVE MONITORING STATUS
# ---------------------------------------------------------

st.subheader("🟢 Live Security Monitoring")

monitor_col1, monitor_col2, monitor_col3 = st.columns(3)

monitor_col1.metric(
    "📡 Monitoring Status",
    "ACTIVE"
)

monitor_col2.metric(
    "📊 Events Loaded",
    len(df)
)

monitor_col3.metric(
    "🕒 Latest Event",
    str(df["timestamp"].max())
)

# =========================================================
# DATA PREPARATION
# =========================================================

# Required columns for the cybersecurity system
required_columns = [
    "timestamp",
    "source_ip",
    "destination_ip",
    "protocol",
    "port",
    "bytes",
    "failed_logins",
    "label"
]

# Check whether required columns exist
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"❌ Missing required columns: {missing_columns}"
    )
    st.stop()

# ---------------------------------------------------------
# REMOVE DUPLICATE RECORDS
# ---------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates().reset_index(drop=True)

duplicates_removed = before_duplicates - len(df)

# ---------------------------------------------------------
# CLEAN ATTACK LABELS
# ---------------------------------------------------------

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
)

df["Attack Type"] = df["label"]

# ---------------------------------------------------------
# CONVERT NUMERIC FEATURES
# ---------------------------------------------------------

numeric_features = [
    "port",
    "bytes",
    "failed_logins"
]

for column in numeric_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ---------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------

missing_before = df[numeric_features].isnull().sum().sum()

for column in numeric_features:

    if df[column].isnull().all():

        df[column] = df[column].fillna(0)

    else:

        df[column] = df[column].fillna(
            df[column].median()
        )

missing_after = df[numeric_features].isnull().sum().sum()

# ---------------------------------------------------------
# CLEAN TIMESTAMP
# ---------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# ---------------------------------------------------------
# SEVERITY CLASSIFICATION
# ---------------------------------------------------------

def get_severity(attack):

    if attack == "Brute Force":
        return "High"

    elif attack == "DNS Attack":
        return "Critical"

    elif attack == "Port Scan":
        return "Medium"

    else:
        return "Low"


df["Severity"] = df["Attack Type"].apply(
    get_severity
)

# ---------------------------------------------------------
# DATA PREPARATION SUMMARY
# ---------------------------------------------------------

st.success(
    f"✅ Data preparation completed successfully. "
    f"{len(df)} clean security events ready for analysis."
)

with st.expander("📋 Data Preparation Details"):

    st.write(
        f"**Total records:** {len(df)}"
    )

    st.write(
        f"**Duplicate records removed:** {duplicates_removed}"
    )

    st.write(
        f"**Missing numeric values handled:** {missing_before}"
    )

    st.write(
        f"**Remaining missing numeric values:** {missing_after}"
    )
# =========================================================
# MACHINE LEARNING
# =========================================================

st.subheader("🤖 AI Threat Detection")

# Features used by the ML model
# ---------------------------------------------------------

# Convert network protocol into numerical values
protocol_mapping = {
    "TCP": 6,
    "UDP": 17,
    "ICMP": 1
}

df["protocol_encoded"] = (
    df["protocol"]
    .astype(str)
    .str.upper()
    .map(protocol_mapping)
    .fillna(0)
)

# Create a traffic-per-login feature
df["bytes_per_login"] = (
    df["bytes"] /
    (df["failed_logins"] + 1)
)

# ML features
features = df[
    [
        "port",
        "bytes",
        "failed_logins",
        "protocol_encoded",
        "bytes_per_login"
    ]
].copy()

# ---------------------------------------------------------
# TRAIN ISOLATION FOREST
# ---------------------------------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.30,
    random_state=42
)

model.fit(features)

# ---------------------------------------------------------
# PREDICT ANOMALIES
# ---------------------------------------------------------

df["ML Prediction"] = model.predict(features)

# Convert prediction into readable result
df["ML Result"] = df["ML Prediction"].apply(
    lambda x: "🚨 Anomaly" if x == -1 else "✅ Normal"
)

# ---------------------------------------------------------
# ANOMALY SCORE
# ---------------------------------------------------------

df["Anomaly Score"] = model.decision_function(features)
# ---------------------------------------------------------
# RISK SCORE
# ---------------------------------------------------------

# Convert anomaly score into a 0-100 risk score
df["Risk Score"] = (
    100 - (
        (df["Anomaly Score"] - df["Anomaly Score"].min())
        /
        (
            df["Anomaly Score"].max()
            - df["Anomaly Score"].min()
            + 0.000001
        )
    ) * 100
)

# Round the score
df["Risk Score"] = df["Risk Score"].round(1)

# Risk classification
def get_risk_level(score):

    if score >= 80:
        return "🔴 Critical"

    elif score >= 60:
        return "🟠 High"

    elif score >= 40:
        return "🟡 Medium"

    else:
        return "🟢 Low"


df["Risk Level"] = df["Risk Score"].apply(
    get_risk_level
)

# Lower score = more unusual event

# ---------------------------------------------------------
# ML STATISTICS
# ---------------------------------------------------------

total_ml_events = len(df)

ml_anomalies = len(
    df[df["ML Prediction"] == -1]
)

ml_normal = len(
    df[df["ML Prediction"] == 1]
)

ml_anomaly_percentage = (
    ml_anomalies / total_ml_events * 100
    if total_ml_events > 0
    else 0
)

# ---------------------------------------------------------
# ML STATUS
# ---------------------------------------------------------

st.success(
    f"🤖 ML model analyzed {total_ml_events} security events "
    f"and detected {ml_anomalies} anomalous events."
)

# ---------------------------------------------------------
# ML PERFORMANCE EVALUATION
# ---------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Ground truth:
# Normal = 0
# Any attack = 1

y_true = df["label"].apply(
    lambda x: 0 if str(x).strip().lower() == "normal" else 1
)

# ML prediction:
# Normal = 0
# Anomaly = 1

y_pred = df["ML Prediction"].apply(
    lambda x: 1 if x == -1 else 0
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

# ---------------------------------------------------------
# DISPLAY ML PERFORMANCE
# ---------------------------------------------------------

st.subheader("📈 ML Model Performance")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Accuracy",
    f"{accuracy * 100:.1f}%"
)

metric2.metric(
    "Precision",
    f"{precision * 100:.1f}%"
)

metric3.metric(
    "Recall",
    f"{recall * 100:.1f}%"
)

metric4.metric(
    "F1 Score",
    f"{f1 * 100:.1f}%"
)

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Attack"],
    columns=["Predicted Normal", "Predicted Anomaly"]
)

st.subheader("📋 Confusion Matrix")

st.dataframe(
    cm_df,
    use_container_width=True
)
# =========================================================
# PHASE 5 - ATTACK CLASSIFICATION
# =========================================================

def classify_attack(row):

    # Normal event
    if row["ML Prediction"] == 1:
        return "Normal"

    # Brute Force detection
    if row["failed_logins"] >= 5:
        return "Brute Force"

    # DNS Attack detection
    if row["port"] == 53 and row["bytes"] >= 5000:
        return "DNS Attack"

    # Port Scan detection
    if row["port"] not in [53, 80, 443] and row["bytes"] < 2000:
        return "Port Scan"

    # Other suspicious activity
    return "Suspicious Activity"


df["Attack Classification"] = df.apply(
    classify_attack,
    axis=1
)
# =========================================================
# PHASE 6 - ALERT MANAGEMENT
# =========================================================

def get_alert_status(row):

    if row["Attack Classification"] == "Normal":
        return "✅ No Alert"

    elif row["Risk Level"] == "🔴 Critical":
        return "🚨 Critical Alert"

    elif row["Risk Level"] == "🟠 High":
        return "⚠️ High Alert"

    elif row["Risk Level"] == "🟡 Medium":
        return "🟡 Medium Alert"

    else:
        return "🟢 Low Alert"


df["Alert Status"] = df.apply(
    get_alert_status,
    axis=1
)
df["Alert Status"] = df.apply(
    get_alert_status,
    axis=1
)

# ==========================================
# SOC SYSTEM STATUS
# ==========================================

st.subheader("🛡️ Security Operations Center")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

status_col1.metric(
    "🟢 System Status",
    "OPERATIONAL"
)

status_col2.metric(
    "📡 Monitoring",
    "ACTIVE"
)

status_col3.metric(
    "🤖 AI Detection",
    "ENABLED"
)

status_col4.metric(
    "🚨 Active Alerts",
    len(df[df["Alert Status"] != "✅ No Alert"])
)
# ---------------------------------------------------------
# RECENT SECURITY ACTIVITY
# ---------------------------------------------------------

st.subheader("📡 Recent Security Activity")

recent_events = df.sort_values(
    by="timestamp",
    ascending=False
).head(5)

recent_display = recent_events[
    [
        "timestamp",
        "source_ip",
        "destination_ip",
        "Attack Classification",
        "Risk Score",
        "Alert Status"
    ]
]

st.dataframe(
    recent_display,
    use_container_width=True,
    hide_index=True
)
# =========================================================
# ALERT OVERVIEW
# =========================================================

st.subheader("🚨 Alert Management")

critical_alerts = len(
    df[df["Alert Status"] == "🚨 Critical Alert"]
)

high_alerts = len(
    df[df["Alert Status"] == "⚠️ High Alert"]
)

medium_alerts = len(
    df[df["Alert Status"] == "🟡 Medium Alert"]
)

total_alerts = len(
    df[df["Alert Status"] != "✅ No Alert"]
)

alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)

alert_col1.metric(
    "🚨 Total Alerts",
    total_alerts
)

alert_col2.metric(
    "🔴 Critical Alerts",
    critical_alerts
)

alert_col3.metric(
    "🟠 High Alerts",
    high_alerts
)

alert_col4.metric(
    "🟡 Medium Alerts",
    medium_alerts
)
active_alerts = df[
    df["Alert Status"] != "✅ No Alert"
].copy()

if len(active_alerts) == 0:

    st.success("✅ No active security alerts.")

else:

    alert_display = active_alerts[
        [
            "timestamp",
            "source_ip",
            "destination_ip",
            "Attack Classification",
            "Severity",
            "Risk Score",
            "Risk Level",
            "Alert Status"
        ]
    ].sort_values(
        by="Risk Score",
        ascending=False
    )

    st.dataframe(
        alert_display,
        use_container_width=True,
        hide_index=True
    )
# =========================================================
# ATTACK CLASSIFICATION RESULTS
# =========================================================

st.subheader("🎯 Attack Classification")

classification_counts = (
    df["Attack Classification"]
    .value_counts()
    .reset_index()
)

classification_counts.columns = [
    "Attack Type",
    "Count"
]

fig_classification = px.bar(
    classification_counts,
    x="Attack Type",
    y="Count",
    title="Classified Security Events",
    text="Count"
)

fig_classification.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_classification,
    use_container_width=True
)

# =========================================================
# AI THREAT DETECTION DETAILS
# =========================================================

st.subheader("🔍 AI Threat Detection Details")

anomaly_df = df[
    df["ML Prediction"] == -1
].copy()
# ---------------------------------------------------------
# AI RISK LEVEL
# ---------------------------------------------------------

def get_ai_risk(score):

    if score <= -0.07:
        return "🔴 Critical"

    elif score <= -0.03:
        return "🟠 High"

    else:
        return "🟡 Medium"


anomaly_df["AI Risk Level"] = anomaly_df[
    "Anomaly Score"
].apply(get_ai_risk)
# ---------------------------------------------------------
# AI ANOMALY EXPLANATION
# ---------------------------------------------------------

feature_medians = df[
    [
        "port",
        "bytes",
        "failed_logins"
    ]
].median()


def explain_anomaly(row):

    reasons = []

    if row["bytes"] > feature_medians["bytes"] * 1.5:
        reasons.append("Unusually high traffic volume")

    if row["failed_logins"] > feature_medians["failed_logins"] + 2:
        reasons.append("High failed login activity")

    if row["port"] not in [22, 53, 80, 443]:
        reasons.append("Unusual network port")

    if len(reasons) == 0:
        reasons.append(
            "Combination of network features differs from normal patterns"
        )

    return " • ".join(reasons)


anomaly_df["AI Explanation"] = anomaly_df.apply(
    explain_anomaly,
    axis=1
)

if len(anomaly_df) == 0:

    st.success("✅ No anomalous events detected by the ML model.")

else:

    st.warning(
        f"🚨 AI detected {len(anomaly_df)} potentially suspicious events."
    )

    anomaly_display = anomaly_df[
        [
            "timestamp",
            "source_ip",
            "destination_ip",
            "protocol",
            "port",
            "bytes",
            "failed_logins",
            "Attack Type",
            "Severity",
            "Anomaly Score",
            "AI Risk Level",
            "AI Explanation"
        ]
    ].sort_values(
        by="Anomaly Score"
    )

    st.dataframe(
        anomaly_display,
        use_container_width=True,
        hide_index=True
    )
# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔎 SOC Filters")

attack_filter = st.sidebar.multiselect(
    "Attack Type",
    options=df["Attack Type"].unique(),
    default=list(df["Attack Type"].unique())
)

severity_filter = st.sidebar.multiselect(
    "Severity",
    options=df["Severity"].unique(),
    default=list(df["Severity"].unique())
)

filtered_df = df[
    (df["Attack Type"].isin(attack_filter)) &
    (df["Severity"].isin(severity_filter))
]

# =========================================================
# METRICS
# =========================================================

total_events = len(filtered_df)

threats = len(
    filtered_df[
        filtered_df["Attack Type"] != "Normal"
    ]
)

high_risk = len(
    filtered_df[
        filtered_df["Severity"] == "High"
    ]
)

critical = len(
    filtered_df[
        filtered_df["Severity"] == "Critical"
    ]
)

ml_anomalies = len(
    filtered_df[
        filtered_df["ML Prediction"] == -1
    ]
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "📡 Total Events",
    total_events
)

col2.metric(
    "🏷️ Labeled Threats",
    threats
)

col3.metric(
    "🔴 High Risk",
    high_risk
)

col4.metric(
    "🔥 Critical",
    critical
)

col5.metric(
    "🤖 ML Anomalies",
    ml_anomalies
)

st.divider()

# =========================================================
# CHARTS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Threat Distribution")

    attack_counts = (
        filtered_df["Attack Type"]
        .value_counts()
        .reset_index()
    )

    attack_counts.columns = [
        "Attack Type",
        "Count"
    ]

    fig_attack = px.bar(
        attack_counts,
        x="Attack Type",
        y="Count",
        title="Detected Security Threats",
        text="Count"
    )

    fig_attack.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_attack,
        use_container_width=True
    )


with col2:

    st.subheader("🎯 Severity Distribution")

    severity_counts = (
        filtered_df["Severity"]
        .value_counts()
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Count"
    ]

    fig_severity = px.pie(
        severity_counts,
        names="Severity",
        values="Count",
        title="Security Severity Levels",
        hole=0.4
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )

# =========================================================
# ML RESULTS
# =========================================================

st.subheader("🤖 Machine Learning Detection")

ml_col1, ml_col2 = st.columns(2)

with ml_col1:

    st.info(
        "The Isolation Forest ML model analyzes network "
        "features such as port, traffic volume and failed "
        "login attempts to identify unusual events."
    )

with ml_col2:

    st.metric(
        "ML Anomalies Detected",
        ml_anomalies
    )

# =========================================================
# ML ANOMALY CHART
# =========================================================

ml_chart = (
    filtered_df["ML Result"]
    .value_counts()
    .reset_index()
)

ml_chart.columns = [
    "ML Result",
    "Count"
]

fig_ml = px.bar(
    ml_chart,
    x="ML Result",
    y="Count",
    title="Machine Learning Anomaly Detection",
    text="Count"
)

st.plotly_chart(
    fig_ml,
    use_container_width=True
)

# =========================================================
# SECURITY EVENTS
# =========================================================

st.subheader("🚨 Security Events")

display_df = filtered_df[
    [
        "timestamp",
        "source_ip",
        "destination_ip",
        "protocol",
        "port",
        "bytes",
        "failed_logins",
        "Attack Type",
        "Attack Classification",
        "Severity",
        "ML Result"
    ]
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# THREAT SUMMARY
# =========================================================

st.subheader("🏷️ Dataset Threat Summary")

threat_counts = (
    filtered_df[
        filtered_df["Attack Type"] != "Normal"
    ]["Attack Type"]
    .value_counts()
)

if len(threat_counts) == 0:

    st.success("✅ No threats detected.")

else:

    for attack_type, count in threat_counts.items():

        if attack_type == "Brute Force":

            st.warning(
                f"⚠️ Brute Force: {count} event(s) detected"
            )

        elif attack_type == "DNS Attack":

            st.error(
                f"🔥 DNS Attack: {count} event(s) detected"
            )

        else:

            st.warning(
                f"⚠️ {attack_type}: {count} event(s) detected"
            )
# =========================================================
# PHASE 8 - SECURITY HISTORY
# =========================================================

st.subheader("📜 Security Event History")

history_col1, history_col2, history_col3 = st.columns(3)

history_col1.metric(
    "📊 Total Events",
    len(df)
)

history_col2.metric(
    "🎯 Classified Threats",
    len(df[df["Attack Classification"] != "Normal"])
)

history_col3.metric(
    "🤖 ML Anomalies",
    len(df[df["ML Prediction"] == -1])
)

# Historical security events
history_display = df.sort_values(
    by="timestamp",
    ascending=False
)[
    [
        "timestamp",
        "source_ip",
        "destination_ip",
        "Attack Classification",
        "Severity",
        "Risk Score",
        "Alert Status"
    ]
]

st.dataframe(
    history_display,
    use_container_width=True,
    hide_index=True
)
# ---------------------------------------------------------
# ATTACK STATISTICS
# ---------------------------------------------------------

st.subheader("📊 Attack Statistics")

attack_statistics = (
    df["Attack Classification"]
    .value_counts()
    .reset_index()
)

attack_statistics.columns = [
    "Attack Type",
    "Number of Events"
]

fig_attack_statistics = px.bar(
    attack_statistics,
    x="Attack Type",
    y="Number of Events",
    title="Attack Classification Statistics",
    text="Number of Events"
)

fig_attack_statistics.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_attack_statistics,
    use_container_width=True
)
# ==========================================
# SECURITY REPORT SUMMARY
# ==========================================

st.subheader("📋 Security Report Summary")

report_col1, report_col2, report_col3, report_col4 = st.columns(4)

report_col1.metric(
    "📊 Total Events",
    len(df)
)

report_col2.metric(
    "🎯 Classified Threats",
    len(df[df["Attack Classification"] != "Normal"])
)

report_col3.metric(
    "🤖 ML Anomalies",
    len(df[df["ML Prediction"] == -1])
)

report_col4.metric(
    "🔴 Critical Alerts",
    len(df[df["Alert Status"] == "🚨 Critical Alert"])
)

st.write("### 🛡️ Threat Distribution")

report_distribution = (
    df["Attack Classification"]
    .value_counts()
    .reset_index()
)

report_distribution.columns = [
    "Attack Type",
    "Number of Events"
]

fig_report = px.pie(
    report_distribution,
    names="Attack Type",
    values="Number of Events",
    title="Security Threat Distribution"
)

st.plotly_chart(
    fig_report,
    use_container_width=True
)
# =========================================================
# DOWNLOAD ALERT REPORT
# =========================================================

st.subheader("📥 Export Security Report")

csv_data = filtered_df.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Alert Report",
    data=csv_data,
    file_name="soc_security_report.csv",
    mime="text/csv"
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Cybersecurity SOC | Machine Learning based anomaly detection"
)