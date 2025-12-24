# ================================
# TASK 1 : ML CLASSIFICATION
# Student Pass / Fail Prediction
# ================================

print("Task 1 started...\n")

# -------- 1. Import libraries --------
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------- 2. Create dataset --------
data = {
    "studytime": [3, 1, 4, 2, 3, 1, 4, 2, 3, 1],
    "absences":  [2,10, 1, 6, 3,12, 0, 7, 4,15],
    "failures":  [0, 1, 0, 1, 0, 2, 0, 1, 0, 2],
    "Result":    ["Pass","Fail","Pass","Fail","Pass","Fail","Pass","Fail","Pass","Fail"]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df, "\n")

# -------- 3. Encode target --------
le = LabelEncoder()
df["Result"] = le.fit_transform(df["Result"])  # Pass=1, Fail=0

# -------- 4. Split features & target --------
X = df[["studytime", "absences", "failures"]]
y = df["Result"]

# -------- 5. Train-test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- 6. Train model --------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------- 7. Predict --------
y_pred = model.predict(X_test)

# -------- 8. Evaluate --------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------- 9. Sample prediction --------
sample = [[3, 4, 0]]  # studytime, absences, failures
result = model.predict(sample)

print("Sample Prediction:", "Pass" if result[0] == 1 else "Fail")

print("\nTask 1 completed successfully.")

input("\nPress Enter to exit...")
