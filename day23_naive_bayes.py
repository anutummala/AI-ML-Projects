import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

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
y = le.fit_transform(df["Category"])  # Minor=0, Young=1, Adult=2
X = df[["Age"]].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)

# -------------------------
# Naive Bayes Model
# -------------------------

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------------
# Visualization (1D Age)
# -------------------------

import numpy as np

age_range = np.linspace(10, 50, 400).reshape(-1, 1)
age_range_scaled = scaler.transform(age_range)
preds = nb.predict(age_range_scaled)

plt.scatter(X, y, c=y, cmap="coolwarm", label="Data")
plt.plot(age_range, preds, color="green", label="Naive Bayes Prediction", linewidth=2)
plt.xlabel("Age")
plt.ylabel("Category (0=Minor,1=Young,2=Adult)")
plt.title("Naive Bayes Decision (1D Age)")
plt.legend()
plt.show()
