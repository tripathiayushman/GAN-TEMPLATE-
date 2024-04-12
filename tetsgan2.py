# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
from tensorflow.keras.datasets import mnist  # You can replace this with your own dataset
import os

# Set the save interval
save_interval = 1000


def save_imgs(epoch):
    r, c = 5, 5  # Number of rows and columns for the grid
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Check if the 'images' directory exists, create it if not
    if not os.path.exists('images'):
        os.makedirs('images')

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()


# Define constants
img_rows = 64
img_cols = 64
channels = 3  # For color images, change to 1 for grayscale
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


# Build the generator model
def build_generator():
    model = Sequential()

    model.add(Dense(128 * 16 * 16, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model


# Build the discriminator model
def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # Adjust the input shape based on the number of channels in your dataset
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# Build and compile the generator
generator = build_generator()
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model, freeze the discriminator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model (generator and discriminator)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Load and normalize the dataset (you can replace this with your own dataset loading and preprocessing)
from tensorflow.keras.datasets import cifar10

# Load and normalize the CIFAR-10 dataset
(X_train, _), (_, _) = cifar10.load_data()
X_train = X_train.astype(np.float32) / 255.0

# If your dataset has a different size or number of channels, adjust img_rows, img_cols, and channels accordingly
img_rows, img_cols, channels = X_train.shape[1:]

# If your dataset is grayscale, reshape it to have a single channel
if channels == 1:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

# Normalize the pixel values to be in the range [-1, 1]
X_train = (X_train - 0.5) * 2.0

# Training the GAN
epochs = 30000
batch_size = 64
half_batch = int(batch_size / 2)

for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    idx = np.random.randint(0, X_train.shape[0], half_batch)
    imgs = X_train[idx]

    noise = np.random.normal(0, 1, (half_batch, latent_dim))

    gen_imgs = generator.predict(noise)

    # Print shapes for debugging
    print("Shape of imgs:", imgs.shape)
    print("Shape of gen_imgs:", gen_imgs.shape)
    print("Shape of real labels:", np.ones((half_batch, 1)).shape)
    print("Shape of fake labels:", np.zeros((half_batch, 1)).shape)

    imgs_resized = resize(imgs, (64, 64))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs_resized, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Train the generator (to have the discriminator label samples as valid)
    g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress
    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    # Save generated images at certain intervals
    if epoch % save_interval == 0:
        save_imgs(epoch)


# Function to save generated images
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()