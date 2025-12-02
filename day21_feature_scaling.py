import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample dataset
data = {
    "Age": [14, 17, 20, 25, 30, 45],
    "Income": [20000, 25000, 30000, 50000, 60000, 80000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# ----------------------
# Min-Max Scaling
# ----------------------
minmax_scaler = MinMaxScaler()
df_minmax = minmax_scaler.fit_transform(df)
print("\nMin-Max Scaled Data:\n", df_minmax)

# ----------------------
# Standardization
# ----------------------
std_scaler = StandardScaler()
df_std = std_scaler.fit_transform(df)
print("\nStandardized Data:\n", df_std)
