# ==================== CELL 1 ====================
# Cell purpose: Imports libraries for data handling, plotting, CNN construction, and train/validation splitting.
import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split


# ==================== CELL 2 ====================
# Cell purpose: Loads Fashion-MNIST locally without downloading anything.
# Change ONLY data_dir to the folder containing your local Fashion-MNIST files.
# Supported formats:
#   1) fashion_mnist.npz / fashion-mnist.npz with x_train, y_train, x_test, y_test
#   2) IDX files, either .gz or uncompressed:
#      train-images-idx3-ubyte, train-labels-idx1-ubyte,
#      t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte

data_dir = r"C:\Users\YourName\Downloads\Fashion-MNIST"

def read_idx_images(file_path):
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic number in {file_path}: {magic}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(count, rows, cols)

def read_idx_labels(file_path):
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic number in {file_path}: {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(count)

def find_file(folder, possible_names):
    for name in possible_names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None

npz_path = find_file(data_dir, [
    "fashion_mnist.npz", "fashion-mnist.npz", "fashionmnist.npz", "mnist.npz"
])

if npz_path is not None:
    with np.load(npz_path) as data:
        required_keys = {"x_train", "y_train", "x_test", "y_test"}
        if not required_keys.issubset(data.files):
            raise ValueError(
                f"{npz_path} must contain x_train, y_train, x_test, and y_test"
            )
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
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
            "Fashion-MNIST files were not found. Set data_dir to the folder "
            "containing fashion_mnist.npz (with x_train/y_train/x_test/y_test) "
            "or the four standard IDX files, optionally compressed with .gz."
        )

    x_train = read_idx_images(train_images_path)
    y_train = read_idx_labels(train_labels_path)
    x_test = read_idx_images(test_images_path)
    y_test = read_idx_labels(test_labels_path)

y_train = np.asarray(y_train).reshape(-1)
y_test = np.asarray(y_test).reshape(-1)

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)


# ==================== CELL 3 ====================
# Cell purpose: Defines the ten Fashion-MNIST class names in label order.
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# ==================== CELL 4 ====================
# Cell purpose: Normalizes pixel values to 0–1 and reshapes grayscale images for the CNN.
# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# ==================== CELL 5 ====================
# Cell purpose: Displays ten sample training images with their class names.
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i],cmap ="gray")
    plt.title(class_names[y_train[i]])
    plt.axis('off')

plt.show()

# ==================== CELL 6 ====================
# Cell purpose: Splits training data into training and validation subsets.
X_train, X_val, y_train_split, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# ==================== CELL 7 ====================
# Cell purpose: Builds a CNN with convolution, pooling, dense, and dropout layers.
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.summary()

# ==================== CELL 8 ====================
# Cell purpose: Compiles the CNN using Adam, sparse categorical cross-entropy, and accuracy.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

# ==================== CELL 9 ====================
# Cell purpose: Trains the CNN for 10 epochs and tracks validation performance.
history = model.fit(
    X_train,
    y_train_split,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val)
)

# ==================== CELL 10 ====================
# Cell purpose: Evaluates the trained CNN on the separate test dataset.
loss, acc = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", acc * 100, "%")

# ==================== CELL 11 ====================
# Cell purpose: Predicts labels for test images and prints the first ten actual and predicted labels.
pred = model.predict(x_test)

pred_classes = np.argmax(pred, axis=1)

print("Actual:", y_test[:10])
print("Predicted:", pred_classes[:10])

# ==================== CELL 12 ====================
# Cell purpose: Displays ten random test images with actual and predicted labels.
plt.figure(figsize=(15, 8))

for i in range(10):
    n = random.randint(0, len(x_test)-1)

    actual = int(y_test[n])
    predicted = int(pred_classes[n])

    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[n].reshape(28, 28), cmap='gray')



    color = "green" if actual == predicted else "red"
    plt.title(f"Actual: {class_names[actual]}\nPredicted: {class_names[predicted]}",
              color=color, fontsize=9)

    plt.axis('off')

plt.tight_layout(pad=2)
plt.show()

# ==================== CELL 13 ====================
# Cell purpose: Shows one test image and prints its actual and predicted class.
n = 10

plt.imshow(x_test[n])
plt.show()

pred = model.predict(x_test[n].reshape(1, 28, 28, 1), verbose=0)

actual = int(y_test[n])
predicted = np.argmax(pred)

print("Actual:", class_names[actual])
print("Predicted:", class_names[predicted])

# ==================== CELL 14 ====================
# Cell purpose: Plots training and validation loss and accuracy.
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.show()

# ==================== CELL 15 ====================
# (This notebook cell is empty.)

# ================== THEORY: FASHION-MNIST CNN CLASSIFICATION ==================
#
# 1. IMAGE CLASSIFICATION
# Image classification assigns an image to one of a fixed set of categories.
# The model learns patterns from labeled examples and predicts labels for new images.
#
# 2. FASHION-MNIST DATASET
# Fashion-MNIST contains 70,000 grayscale clothing images in 10 categories:
# 60,000 training images and 10,000 test images. Each image is 28×28 pixels.
# Labels are integers 0–9, corresponding to the class_names list.
#
# 3. OFFLINE DATA LOADING
# The code reads local NPZ or IDX files and does not download the dataset.
# data_dir must point to the folder containing the supported local files.
#
# 4. NORMALIZATION AND RESHAPING
# Pixel values range from 0 to 255. Dividing by 255 scales them to 0–1.
# The CNN input shape is (number_of_images, 28, 28, 1), with one grayscale channel.
#
# 5. TRAINING AND VALIDATION SPLIT
# train_test_split reserves 20% of the training data for validation.
# Validation data monitors performance but does not update model weights.
# random_state=42 makes the split reproducible.
#
# 6. CONVOLUTIONAL LAYER
# Conv2D applies learnable filters to local image regions to detect features
# such as edges, textures, and clothing parts.
#
# 7. RELU ACTIVATION
# ReLU outputs max(0, x). It adds nonlinearity so the network can learn complex patterns.
#
# 8. MAX POOLING
# MaxPooling2D reduces spatial dimensions by keeping the maximum value in each window.
# This reduces computation while retaining strong detected features.
#
# 9. FLATTEN AND DENSE LAYERS
# Flatten converts feature maps into a vector. Dense layers combine features to
# classify the image into one of ten clothing categories.
#
# 10. DROPOUT
# Dropout randomly disables some units during training and can help reduce overfitting.
#
# 11. SOFTMAX OUTPUT
# The final 10-unit softmax layer produces class probabilities. argmax selects
# the class with the highest score.
#
# 12. LOSS AND OPTIMIZER
# Sparse categorical cross-entropy works with integer labels. Adam updates weights
# using gradients and adaptive learning rates.
#
# 13. EPOCHS AND BATCH SIZE
# An epoch is one pass through the training set. Batch size 128 controls how many
# examples are processed before each weight update.
#
# 14. EVALUATION AND LEARNING CURVES
# Test accuracy is the fraction of test images classified correctly. Training and
# validation curves help identify underfitting and overfitting.
#
# 15. PREDICTION
# model.predict returns ten scores per image. np.argmax selects the largest score,
# and class_names converts its index into a readable class label.
#
# 16. WORKFLOW
# Load local data → normalize and reshape → inspect → split validation data →
# build CNN → compile → train → evaluate → predict → plot learning curves.
#
# ==================== CELL-BY-CELL EXPLANATION ====================
# Cell 1: Imports libraries for data handling, plotting, CNN construction, and splitting.
# Cell 2: Loads Fashion-MNIST locally from NPZ or IDX files without downloading.
# Cell 3: Defines the ten clothing class names.
# Cell 4: Normalizes and reshapes image data for the CNN.
# Cell 5: Displays ten training examples and their labels.
# Cell 6: Creates training and validation subsets.
# Cell 7: Defines the convolutional neural network.
# Cell 8: Compiles the network.
# Cell 9: Trains the network and records history.
# Cell 10: Evaluates performance on test data.
# Cell 11: Prints actual and predicted labels for initial test examples.
# Cell 12: Displays ten random test predictions; green is correct and red is incorrect.
# Cell 13: Displays and predicts one selected test image.
# Cell 14: Plots training and validation loss and accuracy.
