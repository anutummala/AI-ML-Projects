# day10_matplotlib.py
# Python Data Visualization using Matplotlib

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# 1. Load CSV data
df = pd.read_csv("students.csv")

# --------------------------
# 2. Simple bar chart: Number of students per city
city_counts = df["City"].value_counts()
plt.figure(figsize=(6,4))
plt.bar(city_counts.index, city_counts.values, color='skyblue')
plt.title("Number of Students per City")
plt.xlabel("City")
plt.ylabel("Number of Students")
plt.show()

# --------------------------
# 3. Histogram of ages
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=5, color='lightgreen', edgecolor='black')
plt.title("Age Distribution of Students")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# --------------------------
# 4. Pie chart of categories (Minor / Adult)
category_counts = df["Category"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=['pink','lightblue'])
plt.title("Student Category Distribution")
plt.show()
