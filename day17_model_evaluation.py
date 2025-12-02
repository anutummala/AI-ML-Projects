import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, precision_score, accuracy_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pylab as plt

# ----------------------------
# DATASET
# ----------------------------

data = {
    "Name": ["Alice", "Bob", "Bob1", "Bob2", "Bob3", "Bob4"],
    "Age": [22, 25, 25, 19, 14, 25],
    "City": ["New York", "London", "London", "Stockholm", "Paris", "London"]
}

df = pd.DataFrame(data)

# Age → Category
def age_to_category(age):
    if age < 18:
        return "Minor"
    elif age < 30:
        return "Young"
    else:
        return "Adult"
    
df['Category'] = df['Age'].apply(age_to_category)

# ----------------------------
# ENCODING
# ----------------------------

X = df[["Age"]].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Category"])

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)# ----------------------------
# MODEL
# ----------------------------

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------
# METRICS
# ----------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1-score:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------

cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()