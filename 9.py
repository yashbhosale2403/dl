# ==================== CELL 1 ====================
# Cell purpose: Imports libraries and Keras layers used to build the autoencoder.
import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l1


# ==================== CELL 2 ====================
# Cell purpose: Loads MNIST images from local files, normalizes pixels, and flattens images.
# Change ONLY data_dir to your local MNIST dataset folder.
# Supported:
#   - mnist.npz containing x_train, y_train, x_test, y_test
#   - standard IDX image/label files, optionally .gz compressed

data_dir = r"C:\Users\YourName\Downloads\MNIST"

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
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(count)

npz_path = find_file(data_dir, ["mnist.npz", "MNIST.npz"])

if npz_path:
    with np.load(npz_path) as data:
        required = {"x_train", "y_train", "x_test", "y_test"}
        if not required.issubset(data.files):
            raise ValueError("mnist.npz must contain x_train, y_train, x_test, y_test.")
        x_train = data["x_train"]
        x_test = data["x_test"]
else:
    train_images_path = find_file(data_dir, [
        "train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"
    ])
    train_labels_path = find_file(data_dir, [
        "train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"
    ])
    test_images_path = find_file(data_dir, [
        "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"
    ])
    test_labels_path = find_file(data_dir, [
        "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"
    ])

    if any(path is None for path in [
        train_images_path, train_labels_path, test_images_path, test_labels_path
    ]):
        raise FileNotFoundError(
            "MNIST files not found. Set data_dir to a folder containing mnist.npz "
            "or the four standard IDX files (optionally .gz compressed)."
        )

    x_train = read_idx_images(train_images_path)
    x_test = read_idx_images(test_images_path)

# Convert pixel values to 0–1 and flatten each 28×28 image to 784 values.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)


# ==================== CELL 3 ====================
# Cell purpose: Displays ten sample MNIST images to inspect the input data.
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()


# ==================== CELL 4 ====================
# Cell purpose: Builds and compiles an autoencoder that compresses and reconstructs 784-pixel images.
input_layer = Input(shape=(784,))

encoded = Dense(
    128,
    activation="relu",
    activity_regularizer=l1(0.00001)
)(input_layer)

encoded = Dense(64, activation="relu")(encoded)

decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


# ==================== CELL 5 ====================
# Cell purpose: Recompiles the autoencoder with Adam and binary cross-entropy (as in the original notebook).
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


# ==================== CELL 6 ====================
# Cell purpose: Trains the autoencoder to reconstruct its input images for 20 epochs.
history = autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=256,
    validation_data=(x_test, x_test)
)


# ==================== CELL 7 ====================
# Cell purpose: Plots training and validation reconstruction loss over epochs.
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ==================== CELL 8 ====================
# Cell purpose: Shows original test images alongside their reconstructed versions.
decoded_images = autoencoder.predict(x_test[:10], verbose=0)

plt.figure(figsize=(12, 4))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.show()


# ==================== CELL 9 ====================
# Cell purpose: Calculates and prints mean squared reconstruction error for ten test images.
mse = np.mean((x_test[:10] - decoded_images) ** 2, axis=1)

for i, error in enumerate(mse):
    print(f"Image {i + 1}: Reconstruction Error = {error:.5f}")


# ================== THEORY: MNIST DENSE AUTOENCODER ==================
#
# 1. AUTOENCODER
# An autoencoder is a neural network trained to reproduce its input at the output.
# It learns a compact representation of data and then reconstructs the original input.
#
# 2. MNIST DATASET
# MNIST contains grayscale handwritten-digit images from 0 to 9.
# Images are 28×28 pixels, giving 784 pixel values per image.
#
# 3. OFFLINE DATA LOADING
# This practical loads MNIST from a local mnist.npz archive or standard IDX files.
# It does not call mnist.load_data() and does not download the dataset.
#
# 4. NORMALIZATION AND FLATTENING
# Pixel values are converted from 0–255 to 0–1 by dividing by 255.
# Each 28×28 image is flattened into a 784-element vector for Dense layers.
#
# 5. ENCODER
# The encoder maps the 784-dimensional input to 128 units and then to 64 units.
# The 64-unit representation is the compressed latent representation.
#
# 6. DECODER
# The decoder maps the 64-unit representation back through 128 units to 784 outputs.
# The sigmoid output keeps reconstructed pixel values in the range 0–1.
#
# 7. DENSE LAYERS AND RELU
# Dense layers connect each unit to all units in the previous layer.
# ReLU adds nonlinearity and helps the network learn useful representations.
#
# 8. L1 ACTIVITY REGULARIZATION
# The L1 activity regularizer penalizes large encoder activations.
# It encourages a sparse representation, where many activations are near zero.
#
# 9. LOSS FUNCTION
# Binary cross-entropy measures the difference between input pixels and reconstructed
# pixels. The model minimizes this loss during training.
#
# 10. ADAM OPTIMIZER
# Adam updates model parameters using gradients and adaptive learning rates.
#
# 11. TRAINING TARGET
# The input images are also the targets: autoencoder.fit(x_train, x_train).
# The model learns reconstruction rather than digit classification.
#
# 12. RECONSTRUCTION ERROR
# Mean squared error (MSE) averages squared pixel differences between an original
# image and its reconstruction. Lower MSE indicates a closer pixel-wise match.
#
# 13. LOSS CURVES
# Training and validation loss show reconstruction quality over epochs.
# If training loss decreases while validation loss rises, overfitting may be occurring.
#
# 14. WORKFLOW
# Load local MNIST → normalize and flatten → inspect images → build encoder/decoder
# → compile → train on input-as-target → visualize reconstructions → calculate MSE.
#
# ==================== CELL-BY-CELL EXPLANATION ====================
# Cell 1: Imports libraries and autoencoder-related Keras components.
# Cell 2: Loads MNIST offline, normalizes pixels, and flattens images.
# Cell 3: Displays ten sample training images.
# Cell 4: Defines and compiles the encoder-decoder autoencoder.
# Cell 5: Repeats compilation as in the original notebook.
# Cell 6: Trains the autoencoder to reconstruct the input images.
# Cell 7: Plots training and validation loss.
# Cell 8: Compares original images with reconstructed images.
# Cell 9: Prints reconstruction MSE for ten test images.
