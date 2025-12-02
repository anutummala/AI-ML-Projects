import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# Dataset
# -------------------------

data = {
    "Age": [14, 17, 18, 20, 22, 25, 30, 35, 40, 45],
    "Category": ["Minor", "Minor", "Young", "Young", "Young", "Young",
                 "Adult", "Adult", "Adult", "Adult"]
}

df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Category"])
X = df[["Age"]].values

# Scale features (optional for RF, but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)

# -------------------------
# Random Forest Model
# -------------------------

rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------------
# Feature Importance
# -------------------------

import matplotlib.pyplot as plt
import numpy as np

importance = rf.feature_importances_
features = ["Age"]

plt.bar(features, importance)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")
plt.show()
