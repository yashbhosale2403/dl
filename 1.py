# ==================== CELL 1 ====================
# Cell purpose: Imports TensorFlow and NumPy for tensor operations and numerical arrays.
import tensorflow as tf
import numpy as np

# ==================== CELL 2 ====================
# Cell purpose: Creates scalar TensorFlow constants and demonstrates basic arithmetic operations.
a = tf.constant(15)
b = tf.constant(20)

print("Addition:", a+b)
print("Subtraction:", a-b)
print("Multiplication:", a*b)
print("Division:", a/b)

# ==================== CELL 3 ====================
# Cell purpose: Creates a 2D TensorFlow tensor and prints its contents, shape, and data type.
x = tf.constant([[1, 2], [3, 4]])

print("Tensor:\n", x)
print("Shape:", x.shape)
print("Data Type:", x.dtype)

# ==================== CELL 4 ====================
# Cell purpose: Trains a simple linear model y = Wx + b using gradient descent and TensorFlow GradientTape.
x = np.array([1, 2, 3, 4], dtype='float32')
y = 2*x + 1

W = tf.Variable(0.0)
b = tf.Variable(0.0)

for i in range(500):
    with tf.GradientTape() as tape:
        prediction = W*x + b
        loss = tf.reduce_mean((y-prediction)**2)

    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(0.01 * gradients[0])
    b.assign_sub(0.01 * gradients[1])

print("Weight:", W.numpy())
print("Bias:", b.numpy())
print("Prediction:", (W*5+b).numpy())

# ==================== CELL 5 ====================
# Cell purpose: Imports PyTorch for tensor operations and automatic differentiation.
import torch

# ==================== CELL 6 ====================
# Cell purpose: Creates a 2D PyTorch tensor and prints its contents, shape, and data type.
a = torch.tensor([[1, 2], [3, 4]])

print("Tensor:\n", a)
print("Shape:", a.shape)
print("Data Type:", a.dtype)

# ==================== CELL 7 ====================
# Cell purpose: Demonstrates element-wise arithmetic with PyTorch tensors.
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print("Addition:", a+b)
print("Subtraction:", a-b)
print("Multiplication:", a*b)

# ==================== CELL 8 ====================
# Cell purpose: Creates tensors filled with ones, zeros, and random values.
print("Ones:\n", torch.ones(2, 2))
print("Zeros:\n", torch.zeros(2, 2))
print("Random:\n", torch.rand(2, 2))

# ==================== CELL 9 ====================
# Cell purpose: Computes a function and its derivative using PyTorch autograd.
x = torch.tensor(3.0, requires_grad=True)

y = x**2 + 2*x + 1
y.backward()

print("Output:", y.item())
print("Gradient:", x.grad.item())

# ==================== CELL 10 ====================
# Cell purpose: Converts a PyTorch tensor to a NumPy array.
x = torch.tensor([1, 2, 3, 4])

print("PyTorch Tensor:", x)
print("NumPy Array:", x.numpy())

# ================== THEORY: TENSORFLOW AND PYTORCH BASICS ==================
#
# 1. TENSORS
# A tensor is a general data structure for numerical values. Scalars are
# rank-0 tensors, vectors are rank-1 tensors, matrices are rank-2 tensors,
# and higher-dimensional arrays are higher-rank tensors.
#
# 2. TENSORFLOW
# TensorFlow is a machine-learning framework for numerical computation,
# automatic differentiation, and building/training neural networks.
# tf.constant creates an immutable tensor; tf.Variable stores a value that
# can be updated during training.
#
# 3. BASIC ARITHMETIC
# Tensor operations such as addition, subtraction, multiplication, and
# division are performed element-wise when shapes are compatible. TensorFlow
# can also apply operations to scalar tensors.
#
# 4. SHAPE AND DATA TYPE
# Shape describes the dimensions of a tensor. dtype describes the numerical
# type, such as int32 or float32. These properties help ensure operations
# use compatible dimensions and precision.
#
# 5. LINEAR MODEL
# The model prediction is y_hat = Wx + b, where W is the weight and b is the
# bias. The example uses known input-output pairs to learn W and b.
#
# 6. MEAN SQUARED ERROR
# MSE is the average of squared differences between target values and
# predictions. It is a loss function that becomes smaller as predictions
# approach the target values.
#
# 7. GRADIENT DESCENT
# Gradient descent updates parameters in the direction that reduces loss:
# parameter = parameter - learning_rate * gradient.
# The learning rate controls the size of each update.
#
# 8. AUTOMATIC DIFFERENTIATION IN TENSORFLOW
# GradientTape records operations performed inside its context. Calling
# tape.gradient computes derivatives of the loss with respect to selected
# trainable variables, enabling the model to update W and b.
#
# 9. PYTORCH
# PyTorch is another machine-learning framework. torch.tensor creates tensors;
# tensor attributes such as shape and dtype describe their structure.
#
# 10. TENSOR INITIALIZATION
# torch.ones and torch.zeros create tensors filled with ones or zeros.
# torch.rand creates values sampled from a uniform distribution in [0, 1).
#
# 11. AUTOGRAD
# PyTorch tracks operations on tensors with requires_grad=True. Calling
# backward() computes gradients, which are stored in the .grad attribute.
# For y = x^2 + 2x + 1, dy/dx = 2x + 2; at x=3, the gradient is 8.
#
# 12. NUMPY CONVERSION
# A CPU PyTorch tensor can be converted to a NumPy array with .numpy().
# NumPy is useful for numerical analysis and plotting. For tensors on a GPU,
# move them to CPU first using .cpu().numpy().
#
# 13. TENSORFLOW VS PYTORCH
# Both frameworks support tensor computation, GPU acceleration, neural
# networks, and automatic differentiation. Their APIs differ, but the core
# concepts—tensors, operations, gradients, loss, and parameter updates—are
# similar.
#
# 14. WORKFLOW SUMMARY
# Import framework → create tensors → inspect shape and dtype → perform
# operations → compute loss or function → calculate gradients → update
# parameters or inspect results.
#
# ==================== CELL-BY-CELL EXPLANATION ====================
# Cell 1: Imports TensorFlow and NumPy for tensor operations and numerical arrays.
# Cell 2: Creates scalar TensorFlow constants and demonstrates basic arithmetic operations.
# Cell 3: Creates a 2D TensorFlow tensor and prints its contents, shape, and data type.
# Cell 4: Trains a simple linear model y = Wx + b using gradient descent and TensorFlow GradientTape.
# Cell 5: Imports PyTorch for tensor operations and automatic differentiation.
# Cell 6: Creates a 2D PyTorch tensor and prints its contents, shape, and data type.
# Cell 7: Demonstrates element-wise arithmetic with PyTorch tensors.
# Cell 8: Creates tensors filled with ones, zeros, and random values.
# Cell 9: Computes a function and its derivative using PyTorch autograd.
# Cell 10: Converts a PyTorch tensor to a NumPy array.
