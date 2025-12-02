# day9_pandas_basics.py
# Python Pandas Basics for Data Analysis

import pandas as pd

# --------------------------
# 1. Read CSV file
df = pd.read_csv("students.csv")
print("Full DataFrame:\n", df)

# --------------------------
# 2. Access columns
print("\nNames column:\n", df["Name"])
print("\nName and City columns:\n", df[["Name", "City"]])

# --------------------------
# 3. Filter data
london_students = df[df["City"] == "London"]
print("\nStudents in London:\n", london_students)

# --------------------------
# 4. Add a new column (Category based on Age)
def categorize_age(age):
    if age < 18:
        return "Minor"
    else:
        return "Adult"

df["Category"] = df["Age"].apply(categorize_age)
print("\nDataFrame with Category:\n", df)

# --------------------------
# 5. Summary statistics
print("\nAge statistics:")
print(df["Age"].describe())

# --------------------------
# 6. Count by category
print("\nCategory counts:")
print(df["Category"].value_counts())
