# CELL 1: Imports
# NumPy handles arrays, Matplotlib displays images/curves, and TensorFlow builds the neural network.
import os
import gzip
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD


# CELL 2: Load MNIST from your local folder (offline)
# Change ONLY this path to the folder containing your downloaded MNIST files.
# Supported formats:
#   - mnist.npz (with x_train, y_train, x_test, y_test arrays)
#   - IDX files, either .gz compressed or uncompressed:
#       train-images-idx3-ubyte[.gz], train-labels-idx1-ubyte[.gz]
#       t10k-images-idx3-ubyte[.gz],  t10k-labels-idx1-ubyte[.gz]

data_dir = r"C:\Users\YourName\Downloads\MNIST"

def find_local_file(folder, base_name):
    """Find an IDX file whether it is compressed (.gz) or uncompressed."""
    for filename in (base_name, base_name + ".gz"):
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {base_name} or {base_name}.gz in: {folder}"
    )

def open_maybe_gzip(file_path):
    """Open a regular file or a gzip-compressed file in binary mode."""
    if file_path.endswith(".gz"):
        return gzip.open(file_path, "rb")
    return open(file_path, "rb")

def load_idx_images(file_path):
    """Read MNIST image IDX files and return shape (count, 28, 28)."""
    with open_maybe_gzip(file_path) as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image-file magic number {magic}: {file_path}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
    expected = count * rows * cols
    if images.size != expected:
        raise ValueError(
            f"Image file has {images.size} pixels; expected {expected}: {file_path}"
        )
    return images.reshape(count, rows, cols)

def load_idx_labels(file_path):
    """Read MNIST label IDX files and return shape (count,)."""
    with open_maybe_gzip(file_path) as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label-file magic number {magic}: {file_path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(
            f"Label file has {labels.size} labels; expected {count}: {file_path}"
        )
    return labels

# Prefer a local NPZ file if present; otherwise load the four IDX files.
npz_path = os.path.join(data_dir, "mnist.npz")
if os.path.isfile(npz_path):
    with np.load(npz_path) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
else:
    x_train = load_idx_images(find_local_file(data_dir, "train-images-idx3-ubyte"))
    y_train = load_idx_labels(find_local_file(data_dir, "train-labels-idx1-ubyte"))
    x_test = load_idx_images(find_local_file(data_dir, "t10k-images-idx3-ubyte"))
    y_test = load_idx_labels(find_local_file(data_dir, "t10k-labels-idx1-ubyte"))

print("Training Images:", x_train.shape)
print("Training Labels:", y_train.shape)
print("Testing Images:", x_test.shape)
print("Testing Labels:", y_test.shape)


# CELL 3: Normalize pixel values
# Convert integer pixel values (0–255) to float values (0–1).
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# CELL 4: Display ten training images
# Each image is a 28×28 grayscale handwritten digit; its label is 0–9.
plt.figure(figsize=(8, 10))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# CELL 5: Build the neural network
# Flatten turns each 28×28 image into 784 values.
# Two sigmoid hidden layers learn patterns; the output layer has ten digit scores.
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation="sigmoid"),
    Dense(128, activation="sigmoid"),
    Dense(10, activation="sigmoid")
])

model.summary()


# CELL 6: Compile the model
# SGD updates weights using gradients; sparse categorical cross-entropy fits integer labels.
model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# CELL 7: Train the model
# Train for five epochs, using 128 images per batch and 10% of training data for validation.
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128
)


# CELL 8: Evaluate on test data
# Test data is held out from training and estimates performance on unseen digits.
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.2f}%".format(test_acc * 100))


# CELL 9: Predict one test image
# The model expects a batch, so reshape one image to (1, 28, 28).
n = random.randint(0, len(x_test) - 1)

plt.imshow(x_test[n], cmap="gray")
plt.axis("off")
plt.show()

pred = model.predict(x_test[n].reshape(1, 28, 28), verbose=0)

print("Actual:", y_test[n])
print("Predicted:", np.argmax(pred))


# CELL 10: Plot training history
# Compare training and validation loss/accuracy across epochs.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()


# ==================== THEORY: MNIST DIGIT CLASSIFICATION ====================
#
# 1. MNIST DATASET
# MNIST contains grayscale images of handwritten digits from 0 to 9.
# It has 60,000 training images and 10,000 test images, each 28×28 pixels.
# Each image has an integer label identifying the correct digit.
#
# 2. OFFLINE DATA LOADING
# The dataset is read from files already stored on the computer, so the
# practical does not need an internet connection or keras.datasets download.
# MNIST may be stored as mnist.npz or as IDX image/label files, optionally gzipped.
#
# 3. IMAGE ARRAYS AND LABELS
# An image array has shape (number_of_images, 28, 28). Labels are integers 0–9.
# Images contain grayscale intensity values from 0 to 255.
#
# 4. NORMALIZATION
# Dividing pixel values by 255 scales them to 0–1. Smaller consistent inputs
# usually make neural-network training more stable.
#
# 5. FLATTEN LAYER
# Flatten converts each 28×28 image into a one-dimensional vector of 784 values.
# It changes the shape but does not learn weights.
#
# 6. DENSE LAYERS AND SIGMOID
# Dense layers connect every neuron to all values from the previous layer.
# Sigmoid maps values into the range 0–1 and introduces nonlinearity.
#
# 7. OUTPUT LAYER
# The final Dense layer has 10 neurons, one for each digit. The largest output
# score is treated as the predicted digit using argmax.
#
# 8. LOSS AND OPTIMIZER
# Sparse categorical cross-entropy compares predictions with integer class labels.
# SGD (stochastic gradient descent) adjusts model weights to reduce this loss.
#
# 9. EPOCH, BATCH SIZE, AND VALIDATION
# An epoch is one full pass through the training data. Batch size 128 means
# weights are updated after processing batches of 128 images. Validation_split
# reserves 10% of training data to monitor performance without updating weights.
#
# 10. TESTING AND ACCURACY
# Evaluation uses the separate test set. Accuracy is the fraction of digits
# classified correctly; loss measures prediction error.
#
# 11. LEARNING CURVES
# Training and validation curves show how model performance changes over epochs.
# A widening gap can indicate overfitting.
#
# 12. WORKFLOW
# Load local data → normalize → inspect samples → build model → compile →
# train → evaluate → predict a sample → plot training history.

