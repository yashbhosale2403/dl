# Denoising Autoencoder for MNIST
# Converted from DL_Chit_11.ipynb into a cell-wise Python script.
# Run cells in order. Change data_dir in CELL 2 to your local MNIST folder.

# ==================== CELL 1 ====================
# Cell purpose: Imports libraries and neural-network components.
import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# ==================== CELL 2 ====================
# Cell purpose: Loads MNIST from local files; no internet download is performed.
# Change ONLY data_dir. Supported formats:
#   - mnist.npz with x_train, y_train, x_test, y_test
#   - standard IDX image/label files, optionally .gz compressed

data_dir = r"C:\\Users\\YourName\\Downloads\\MNIST"

def find_file(folder, names):
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None

def read_idx_images(file_path):
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image file: {file_path}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(count, rows, cols)

def read_idx_labels(file_path):
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label file: {file_path}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(count)

npz_path = find_file(data_dir, ["mnist.npz", "MNIST.npz"])
if npz_path:
    with np.load(npz_path) as data:
        required = {"x_train", "y_train", "x_test", "y_test"}
        if not required.issubset(data.files):
            raise ValueError("MNIST NPZ must contain x_train, y_train, x_test, y_test.")
        x_train, x_test = data["x_train"], data["x_test"]
else:
    paths = [
        find_file(data_dir, ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"]),
        find_file(data_dir, ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"]),
        find_file(data_dir, ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"]),
        find_file(data_dir, ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"]),
    ]
    if any(path is None for path in paths):
        raise FileNotFoundError(
            "MNIST files not found. Set data_dir to mnist.npz or the four IDX files."
        )
    x_train = read_idx_images(paths[0])
    x_test = read_idx_images(paths[2])

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)


# ==================== CELL 3 ====================
# Cell purpose: Adds random noise to images and clips pixel values to the valid range.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# ==================== CELL 4 ====================
# Cell purpose: Displays clean and noisy training images for comparison.
noise_factor = 0.4

x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)

# ==================== CELL 5 ====================
# Cell purpose: Builds the encoder-decoder network that learns to remove noise.
plt.figure(figsize=(12, 4))

for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 10, i+11)
    plt.imshow(x_train_noisy[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()

# ==================== CELL 6 ====================
# Cell purpose: Compiles the model using Adam and binary cross-entropy.
input_layer = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.summary()

# ==================== CELL 7 ====================
# Cell purpose: Trains the network to map noisy images to clean images.
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# ==================== CELL 8 ====================
# Cell purpose: Plots training and validation loss during denoising.
history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=20,
    batch_size=256,
    validation_data=(x_test_noisy, x_test)
)

# ==================== CELL 9 ====================
# Cell purpose: Compares clean, noisy, and reconstructed test images.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Denoising Autoencoder Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ==================== CELL 10 ====================
# Cell purpose: Calculates reconstruction mean squared error for ten test images.
denoised = autoencoder.predict(x_test_noisy[:10])

plt.figure(figsize=(12, 4))

for i in range(10):
    plt.subplot(3, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 10, i+11)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 10, i+21)
    plt.imshow(denoised[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()

# ==================== CELL 11 ====================
# Cell purpose: Executes the original notebook cell.
mse = np.mean((x_test[:10] - denoised) ** 2, axis=1)

for i, error in enumerate(mse):
    print(f"Image {i+1}: MSE = {error:.5f}")

# ==================== THEORY / VIVA NOTES ====================
# 1. Autoencoder:
#    A neural network trained to reproduce its input. It contains an encoder,
#    a compressed representation (latent space), and a decoder.
#
# 2. Denoising autoencoder:
#    The input is intentionally corrupted, while the target remains clean.
#    The model learns to recover useful structure instead of copying noise.
#
# 3. MNIST:
#    A dataset of handwritten digit images, each 28x28 grayscale pixels.
#    Flattening converts each image into a vector of 784 pixel values.
#
# 4. Noise generation:
#    Gaussian noise is added using noise_factor * random_noise.
#    np.clip(..., 0, 1) keeps pixel values in the normalized range.
#
# 5. Network architecture:
#    Encoder: 784 -> 128 -> 64 neurons.
#    Decoder: 64 -> 128 -> 784 neurons.
#    ReLU learns nonlinear features; sigmoid outputs values from 0 to 1.
#
# 6. Training:
#    Noisy images are inputs and clean images are targets.
#    Adam updates weights; binary cross-entropy measures reconstruction loss.
#    Epochs are full passes through training data; batch_size controls
#    examples processed before each weight update.
#
# 7. Validation loss:
#    Evaluates reconstruction on unseen test images during training.
#    A widening gap between training and validation loss can indicate overfit.
#
# 8. MSE:
#    Mean squared error averages squared pixel differences.
#    Lower MSE generally means the reconstructed image is closer to the clean
#    target, though visual quality and other metrics can also matter.
