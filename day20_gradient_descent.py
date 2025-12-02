import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Dataset
# -------------------------

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)  # perfect line: y = 2x + 1

# Convert to 2D for consistency
X = X.reshape(-1, 1)

# -------------------------
# Step 1: Initialize parameters
# -------------------------

m = 0.0
b = 0.0

learning_rate = 0.01
epochs = 500

losses = []

# -------------------------
# Step 2: Training Loop
# -------------------------

for i in range(epochs):

    y_pred = m * X + b

    error = y_pred - y
    loss = np.mean(error ** 2)
    losses.append(loss)

    dm = np.mean(2 * error * X)
    db = np.mean(2 * error)

    m -= learning_rate * dm
    b -= learning_rate * db

    if i % 50 == 0:
        print(f"Epoch {i}: Loss={loss:.4f}, m={m:.4f}, b={b:.4f}")

# -------------------------
# Final results
# -------------------------

print("\nTraining complete!")
print(f"Final slope (m): {m}")
print(f"Final intercept (b): {b}")

# -------------------------
# Plot loss curve
# -------------------------

plt.plot(losses)
plt.title("Loss Curve (Gradient Descent)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# -------------------------
# Prediction Example
# -------------------------

test_value = 6
y_test_pred = m * test_value + b
print(f"Prediction for x={test_value}: {y_test_pred}")
