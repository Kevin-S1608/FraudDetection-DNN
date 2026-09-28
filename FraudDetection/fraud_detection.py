import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("fraud_transactions.csv")

print("Dataset Shape:", data.shape)
print("Legitimate Transactions:", (data["Class"] == 0).sum())
print("Fraudulent Transactions:", (data["Class"] == 1).sum())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop("Class", axis=1)
y = data["Class"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for website
joblib.dump(scaler, "scaler.pkl")

print("\nScaler saved as scaler.pkl")


# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    zip(np.unique(y_train), weights)
)

print("Class Weights:", class_weights)


# ==========================================
# 6. BUILD DEEP NEURAL NETWORK
# ==========================================

model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(16, activation="relu"),

    Dense(8, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Deep Neural Network...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)

print("\nTraining completed!")


# ==========================================
# 9. SAVE TRAINED MODEL
# ==========================================

model.save("fraud_model.keras")

print("Model saved as fraud_model.keras")


# ==========================================
# 10. MODEL PREDICTION
# ==========================================

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_prediction = (y_probability >= 0.5).astype(int)


# ==========================================
# 11. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)


print("\n================================")
print("       MODEL EVALUATION")
print("================================")

print(
    "Accuracy  :",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision :",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall    :",
    round(recall * 100, 2),
    "%"
)

print(
    "F1-Score  :",
    round(f1 * 100, 2),
    "%"
)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_prediction
)

print("\nConfusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Legitimate",
        "Fraudulent"
    ]
)

display.plot()

plt.title("Confusion Matrix")

plt.show()


# ==========================================
# 13. MANUAL TRANSACTION TEST
# ==========================================

print("\n================================")
print("     FRAUD DETECTION SYSTEM")
print("================================")

print("\nEnter Transaction Details:")

time = float(input("Time: "))
amount = float(input("Amount: "))
v1 = float(input("V1: "))
v2 = float(input("V2: "))
v3 = float(input("V3: "))
v4 = float(input("V4: "))
v5 = float(input("V5: "))
v6 = float(input("V6: "))


# Create transaction DataFrame
transaction = pd.DataFrame(
    [[
        time,
        amount,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6
    ]],
    columns=X.columns
)


# Scale transaction
transaction_scaled = scaler.transform(
    transaction
)


# Predict
probability = model.predict(
    transaction_scaled,
    verbose=0
)[0][0]


# ==========================================
# 14. DISPLAY RESULT
# ==========================================

print("\n================================")

if probability >= 0.5:

    print(
        "Prediction : FRAUDULENT TRANSACTION"
    )

else:

    print(
        "Prediction : LEGITIMATE TRANSACTION"
    )


print(
    "Fraud Probability :",
    round(float(probability) * 100, 2),
    "%"
)

print("================================")