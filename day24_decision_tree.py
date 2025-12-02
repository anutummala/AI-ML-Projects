import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
y = le.fit_transform(df["Category"])
X = df[["Age"]].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# -------------------------
# Decision Tree Model
# -------------------------

dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------------
# Visualization
# -------------------------

plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=["Age"], class_names=le.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
