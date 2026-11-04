# ==================== CELL 1 ====================
# Cell purpose: Imports NumPy, Matplotlib, TensorFlow, the CIFAR-10 dataset, Keras Sequential, and Dense/Flatten layers.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# ==================== CELL 2 ====================
# Cell purpose: Loads CIFAR-10 training and test images from the local dataset folder.
# Change only data_dir to the location of your extracted cifar-10-batches-py folder.

data_dir = r"C:\Users\YourName\Downloads\cifar-10-batches-py"

def load_cifar_batch(file_path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    labels = np.array(batch[b"labels"])

    return images, labels

# Load the five training batches
x_train_list = []
y_train_list = []

for i in range(1, 6):
    file_path = os.path.join(data_dir, f"data_batch_{i}")
    images, labels = load_cifar_batch(file_path)
    x_train_list.append(images)
    y_train_list.append(labels)

x_train = np.concatenate(x_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0).reshape(-1, 1)

# Load the test batch
test_path = os.path.join(data_dir, "test_batch")
x_test, y_test = load_cifar_batch(test_path)
y_test = y_test.reshape(-1, 1)

print("Training data:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing data:", x_test.shape)
print("Testing labels:", y_test.shape)

# ==================== CELL 3 ====================
# Cell purpose: Defines the ten CIFAR-10 class names in the dataset's label order.
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("Classes:", class_names)

# ==================== CELL 4 ====================
# Cell purpose: Displays ten training images in a 2×5 grid with their class names.
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i][0])])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ==================== CELL 5 ====================
# Cell purpose: Displays one selected training image and its label.

plt.imshow(x_train[1])
plt.title(class_names[int(y_train[1][0])])

# ==================== CELL 6 ====================
# Cell purpose: Converts pixel values to float32 and normalizes them from 0–255 to 0–1.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# ==================== CELL 7 ====================
# Cell purpose: Builds a fully connected neural network that flattens each 32×32 RGB image and uses two ReLU hidden layers and a 10-class softmax output.
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# ==================== CELL 8 ====================
# Cell purpose: Compiles the model with Adam, sparse categorical cross-entropy, and accuracy.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== CELL 9 ====================
# Cell purpose: Trains for 10 epochs with batch size 64 and 10% validation data.
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# ==================== CELL 10 ====================
# Cell purpose: Evaluates the trained model on the test set and prints test loss and accuracy.
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# ==================== CELL 11 ====================
# Cell purpose: Plots training and validation accuracy and loss across epochs.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==================== CELL 12 ====================
# Cell purpose: Chooses a random test image, predicts its class, and displays actual and predicted labels.
n = np.random.randint(0, len(x_test))

prediction = model.predict(x_test[n].reshape(1, 32, 32, 3), verbose=0)

plt.imshow(x_test[n])
plt.title("Actual: " + class_names[int(y_test[n][0])] +
          "\nPredicted: " + class_names[np.argmax(prediction)])
plt.axis('off')
plt.show()

# ==================== CELL 13 ====================
# (This notebook cell is empty.)
# (This notebook cell is empty.)

# ==================== CELL 14 ====================
# (This notebook cell is empty.)
# (This notebook cell is empty.)

# ==================== CELL 15 ====================
# (This notebook cell is empty.)
# (This notebook cell is empty.)

# ==================== CELL 16 ====================
# (This notebook cell is empty.)
# (This notebook cell is empty.)

# ================== THEORY: CIFAR-10 IMAGE CLASSIFICATION ==================
#
# 1. IMAGE CLASSIFICATION
# Image classification is the task of assigning an image to one category.
# The model learns visual patterns from labelled examples and predicts a class
# for images it has not seen during training.
#
# 2. CIFAR-10 DATASET
# CIFAR-10 contains 60,000 color images across 10 classes: airplane,
# automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The standard
# split has 50,000 training images and 10,000 test images. Each image is
# 32×32 pixels with 3 color channels (RGB). Each label is an integer 0–9.
#
# 3. RGB IMAGES AND IMAGE SHAPE
# Unlike grayscale images, color images store three values per pixel:
# red, green, and blue. Therefore an image has shape (32, 32, 3): height,
# width, and channels. A batch adds another dimension for the number of images.
#
# 4. DATA EXPLORATION AND LABELS
# Displaying sample images helps verify that the dataset loaded properly and
# that numeric labels map to the correct class names. CIFAR-10 labels are
# integer IDs; class_names provides human-readable names in the same order.
#
# 5. NORMALIZATION
# Pixel values range from 0 to 255. Casting to float32 and dividing by 255
# scales values to 0–1. This consistent, smaller range generally helps
# optimization and numerical stability.
#
# 6. FLATTEN AND DENSE NETWORK
# Flatten converts the 32×32×3 image (3072 values) into one vector. Dense
# layers connect each neuron to every input value. This network uses 512 and
# 256 hidden neurons, followed by 10 output neurons. Flatten does not learn
# parameters; Dense layers learn weights and biases.
#
# 7. RELU ACTIVATION
# ReLU is f(z)=max(0,z): negative values become zero, positive values remain.
# It adds nonlinearity so the network can learn complex patterns and is often
# easier to optimize than sigmoid in hidden layers.
#
# 8. SOFTMAX OUTPUT
# Softmax converts the ten output scores into values between 0 and 1 that sum
# to 1. They can be interpreted as class probabilities. The largest value
# corresponds to the model's predicted class.
#
# 9. SPARSE CATEGORICAL CROSS-ENTROPY
# This loss is suitable when each example belongs to exactly one class and
# labels are stored as integer IDs rather than one-hot vectors. It penalizes
# predictions that give low probability to the correct class.
#
# 10. ADAM OPTIMIZER
# Adam updates weights using gradients while adapting step sizes for individual
# parameters based on moving averages of gradients and squared gradients.
# It often works well without much manual learning-rate tuning.
#
# 11. EPOCHS, BATCH SIZE, AND VALIDATION
# An epoch is one pass through the training data. Batch size 64 means the
# model processes 64 images before each update. The validation_split=0.1
# setting reserves 10% of training data to monitor performance during training;
# validation examples do not update the weights.
#
# 12. TESTING AND METRICS
# Test data is held out from training and used to estimate performance on new
# images. Accuracy is the fraction classified correctly. Loss measures the
# confidence-weighted prediction error and is used for optimization.
#
# 13. LEARNING CURVES AND OVERFITTING
# Training curves show performance on training examples; validation curves
# show performance on held-out validation examples. If training accuracy keeps
# rising while validation accuracy stalls or drops, or validation loss rises,
# the model may be overfitting.
#
# 14. MAKING A PREDICTION
# The model expects a batch-shaped input, so reshape one image to
# (1, 32, 32, 3). predict returns ten output scores. argmax finds the index of
# the largest score, and class_names translates that index into a class name.
#
# 15. LIMITATION OF A FLATTEN-DENSE MODEL
# Flatten removes the explicit 2D neighborhood structure before learning.
# Dense layers can learn useful patterns, but they do not naturally exploit
# nearby pixels or translation as convolutional neural networks do. CNNs are
# therefore commonly used for image classification, especially for richer images.
#
# 16. COMPLETE WORKFLOW
# Load images and labels → inspect samples → normalize pixels → define network
# → compile with optimizer/loss/metric → train → evaluate on test data →
# plot learning curves → predict a sample image.
#
# ==================== CELL-BY-CELL EXPLANATION ====================
# Cell 1: Imports NumPy, Matplotlib, TensorFlow, the CIFAR-10 dataset, Keras Sequential, and Dense/Flatten layers.
# Cell 2: Loads CIFAR-10 training and test images with labels and prints their shapes.
# Cell 3: Defines the ten CIFAR-10 class names in the dataset's label order.
# Cell 4: Displays ten training images in a 2×5 grid with their class names.
# Cell 5: Displays one selected training image and its label.
# Cell 6: Converts pixel values to float32 and normalizes them from 0–255 to 0–1.
# Cell 7: Builds a fully connected neural network that flattens each 32×32 RGB image and uses two ReLU hidden layers and a 10-class softmax output.
# Cell 8: Compiles the model with Adam, sparse categorical cross-entropy, and accuracy.
# Cell 9: Trains for 10 epochs with batch size 64 and 10% validation data.
# Cell 10: Evaluates the trained model on the test set and prints test loss and accuracy.
# Cell 11: Plots training and validation accuracy and loss across epochs.
# Cell 12: Chooses a random test image, predicts its class, and displays actual and predicted labels.
