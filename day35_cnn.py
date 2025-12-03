import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -------------------------
# Load MNIST Dataset
# -------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape for CNN: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------
# Build CNN Model
# -------------------------
model = Sequential()

# Convolution + Pooling
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten + Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# -------------------------
# Compile Model
# -------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------
# Train Model
# -------------------------
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128)

# -------------------------
# Evaluate Model
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
