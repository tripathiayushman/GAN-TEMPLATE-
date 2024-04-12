# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Define the generator model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(784, activation='sigmoid'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Define the discriminator model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Load and preprocess the dataset (MNIST in this example)
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
train_images = np.expand_dims(train_images, axis=-1)

# Build and compile the discriminator
img_shape = train_images[0].shape
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build and compile the generator
latent_dim = 100
generator = build_generator(latent_dim)

# Build and compile the GAN model
discriminator.trainable = False
gan = build_gan(generator, discriminator)

# Training the GAN
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]
    real_labels = np.ones((batch_size, 1))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    generated_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_images, generated_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress and save generated images at certain intervals
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        # Save generated images
        generated_images = generator.predict(noise)
        generated_images = generated_images * 0.5 + 0.5  # Rescale to [0, 1]
        fig, axs = plt.subplots(2, 5)
        count = 0
        for i in range(2):
            for j in range(5):
                axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.show()
