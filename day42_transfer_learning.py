import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# ------------------------------------
# Load CIFAR-10
# ------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

# Choose two classes: cats=3, dogs=5
classes = [3, 5]

# Create masks
train_mask = np.isin(y_train, classes)
test_mask = np.isin(y_test, classes)

# Apply masks
X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]


# Convert labels: cat=0, dog=1
y_train = (y_train == 5).astype(int)
y_test = (y_test == 5).astype(int)

# One-hot encode
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Resize CIFAR-10 (32x32 → 224x224)
X_train = tf.image.resize(X_train, (224, 224)) / 255.0
X_test  = tf.image.resize(X_test, (224, 224)) / 255.0

# ------------------------------------
# PRETRAINED VGG16
# ------------------------------------
vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
vgg.trainable = False   # freeze feature extractor

model = Sequential([
    vgg,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ------------------------------------
# Train
# ------------------------------------
history = model.fit(X_train, y_train, epochs=3, batch_size=32)

# ------------------------------------
# Evaluate
# ------------------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy: ", acc)