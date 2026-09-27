import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. CREATE SYNTHETIC CYBERSECURITY TRAINING DATA
# =========================================================

np.random.seed(42)

n = 1200

attack_types = np.random.choice(
    [
        "Normal",
        "Brute Force",
        "DNS Attack",
        "Port Scan"
    ],
    size=n,
    p=[0.45, 0.25, 0.15, 0.15]
)

rows = []

for attack in attack_types:

    if attack == "Normal":

        protocol = np.random.choice(["TCP", "UDP"])
        port = np.random.choice([80, 443, 8080])
        bytes_sent = np.random.randint(500, 5000)
        failed_logins = np.random.randint(0, 2)

    elif attack == "Brute Force":

        protocol = "TCP"
        port = 22
        bytes_sent = np.random.randint(100, 1000)
        failed_logins = np.random.randint(5, 20)

    elif attack == "DNS Attack":

        protocol = "UDP"
        port = 53
        bytes_sent = np.random.randint(5000, 20000)
        failed_logins = 0

    else:

        protocol = "TCP"
        port = np.random.choice(
            [20, 21, 22, 23, 25, 80, 443]
        )
        bytes_sent = np.random.randint(50, 1500)
        failed_logins = np.random.randint(0, 3)

    rows.append(
        [
            protocol,
            port,
            bytes_sent,
            failed_logins,
            attack
        ]
    )


dataset = pd.DataFrame(
    rows,
    columns=[
        "protocol",
        "port",
        "bytes",
        "failed_logins",
        "label"
    ]
)

dataset.to_csv(
    "data/ml_training_data.csv",
    index=False
)

print("✅ Training dataset created.")
print(f"📊 Total records: {len(dataset)}")

# =========================================================
# 2. FEATURES AND TARGET
# =========================================================

X = dataset[
    [
        "protocol",
        "port",
        "bytes",
        "failed_logins"
    ]
]

y = dataset["label"]

# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# 4. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "protocol",
            OneHotEncoder(handle_unknown="ignore"),
            ["protocol"]
        )
    ],
    remainder="passthrough"
)

# =========================================================
# 5. RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# =========================================================
# 6. TRAIN MODEL
# =========================================================

pipeline.fit(
    X_train,
    y_train
)

print("✅ Random Forest model trained.")

# =========================================================
# 7. EVALUATE MODEL
# =========================================================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# =========================================================
# 8. SAVE MODEL
# =========================================================

joblib.dump(
    pipeline,
    "cybersecurity_model.pkl"
)

print("\n✅ Model saved as cybersecurity_model.pkl")

print(
    "\n🚀 Phase 3 model training completed successfully!"
)