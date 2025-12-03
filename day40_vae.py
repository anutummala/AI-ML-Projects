import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# -------------------------
# Load MNIST
# -------------------------
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))
input_dim = X_train.shape[1]

# -------------------------
# VAE Parameters
# -------------------------
latent_dim = 2
hidden_dim = 128

# -------------------------
# Sampling Function
# -------------------------
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# -------------------------
# Encoder
# -------------------------
inputs = Input(shape=(input_dim,))
h = Dense(hidden_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_var])

# -------------------------
# Decoder
# -------------------------
decoder_h = Dense(hidden_dim, activation='relu')
decoder_out = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

# -------------------------
# VAE Model
# -------------------------
vae = Model(inputs, outputs)

# VAE Loss = Reconstruction + KL Divergence
reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= input_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1) * -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# -------------------------
# Train VAE
# -------------------------
vae.fit(X_train, epochs=50, batch_size=256, validation_data=(X_test, None), verbose=0)

# -------------------------
# Encode & Visualize
# -------------------------
encoder = Model(inputs, z_mean)
z_test = encoder.predict(X_test)

plt.scatter(z_test[:,0], z_test[:,1])
plt.xlabel("z1")
plt.ylabel("z2")
plt.title("2D Latent Space of VAE")
plt.show()
