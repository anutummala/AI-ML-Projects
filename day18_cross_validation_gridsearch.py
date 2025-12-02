import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# Sample dataset
# ---------------------------

data = {
    "Name": ["Alice", "Bob", "Bob1", "Bob2", "Bob3", "Bob4"],
    "Age": [22, 25, 25, 19, 14, 25],
    "City": ["New York", "London", "London", "Stockholm", "Paris", "London"]
}

df = pd.DataFrame(data)

def age_to_category(age):
    if age < 18:
        return "Minor"
    elif age < 30:
        return "Young"
    else:
        return "Adult"
    
df['Category'] = df['Age'].apply(age_to_category)

# Input & label encoding
X = df[["Age"]].values
le = LabelEncoder()
y = le.fit_transform(df["Category"])


# ---------------------------
# 1. Simple Train/Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Train/Test Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 2. 5-fold Cross Validation
# ---------------------------

cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))

# ---------------------------
# 3. Hyperparameter Tuning
# ---------------------------

param_grid = {
    "max_depth": [1, 2, 3, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 3, 4]
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X, y)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
print("\nBest Model:", grid.best_estimator_)