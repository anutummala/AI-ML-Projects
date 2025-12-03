import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

# -------------------------
# Load MNIST
# -------------------------
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.astype('float32') / 127.5 - 1  # Normalize to [-1,1]
X_train = X_train.reshape((X_train.shape[0], -1))

# -------------------------
# GAN Parameters
# -------------------------
img_dim = X_train.shape[1]  # 28*28 = 784
latent_dim = 100
batch_size = 128
epochs = 10000

# -------------------------
# Build Generator
# -------------------------
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(img_dim, activation='tanh'))
    return model

# -------------------------
# Build Discriminator
# -------------------------
def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# -------------------------
# Compile Models
# -------------------------
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()
z = np.random.normal(0, 1, (batch_size, latent_dim))
discriminator.trainable = False

# Combined model (Generator + Discriminator)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
latent_input = Input(shape=(latent_dim,))
img = generator(latent_input)
validity = discriminator(img)
combined = Model(latent_input, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epochs):
    # ---------------------
    # Train Discriminator
    # ---------------------
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size,1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size,1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    # Train Generator
    # ---------------------
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, np.ones((batch_size,1)))

    # Print progress
    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")

# -------------------------
# Generate Images
# -------------------------
noise = np.random.normal(0,1,(10,latent_dim))
gen_imgs = generator.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0,1]

plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(gen_imgs[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()
