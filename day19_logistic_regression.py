import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Dataset
# -------------------------

data = {
    "Age": [14, 17, 18, 20, 22, 25, 30, 35, 40, 45],
    "Label": ["Minor", "Minor", "Young", "Young", "Young", "Young",
              "Adult", "Adult", "Adult", "Adult"]
}

df = pd.DataFrame(data)

# Encode labels (Minor=0, Young=1, Adult=2)
encoder = LabelEncoder()
y = encoder.fit_transform(df["Label"])
X = df[["Age"]].values  # shape (n_samples, 1)

# Convert this to a binary classification: Minor(0) vs Not-Minor(1)
y_binary = (df["Label"] != "Minor").astype(int)  # Minor=0, Young/Adult=1

# Train logistic regression
model = LogisticRegression()
model.fit(X, y_binary)

# -------------------------
# Decision Boundary
# -------------------------

# Create points from age 10 to 50 for smooth curve
age_range = np.linspace(10, 50, 200).reshape(-1, 1)
probabilities = model.predict_proba(age_range)[:, 1]

# Plot data
plt.scatter(X, y_binary, color="blue", label="Data (0=Minor, 1=Not Minor)")

# Plot sigmoid curve
plt.plot(age_range, probabilities, color="red", label="Sigmoid Curve")

# Decision boundary at probability = 0.5
plt.axhline(0.5, color="green", linestyle="--")
plt.text(10, 0.52, "Decision Boundary (0.5)", color="green")

plt.xlabel("Age")
plt.ylabel("Probability (Not Minor)")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()

# -------------------------
# Testing the model
# -------------------------

test_ages = np.array([[16], [19], [29], [50]])
preds = model.predict(test_ages)
pred_probs = model.predict_proba(test_ages)[:, 1]

for age, p, prob in zip(test_ages, preds, pred_probs):
    print(f"Age {age[0]} → Predicted: {p} → Prob: {prob:.2f}")
