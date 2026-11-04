# ============== CELL 1 ==============
# Cell purpose: Imports TensorFlow for tensor operations and NumPy for numerical arrays.
import tensorflow as tf
import numpy as np

# ============== CELL 2 ==============
# Cell purpose: Creates scalar TensorFlow constants and demonstrates arithmetic operations.
a = tf.constant(15)
b = tf.constant(20)

print("Addition:", a+b)
print("Subtraction:", a-b)
print("Multiplication:", a*b)
print("Division:", a/b)

# ============== CELL 3 ==============
# Cell purpose: Creates a 2×2 tensor and prints its values, shape, and data type.
x = tf.constant([[1, 2], [3, 4]])

print("Tensor:\n", x)
print("Shape:", x.shape)
print("Data Type:", x.dtype)

# ============== CELL 4 ==============
# Cell purpose: Trains a simple linear model y = Wx + b using gradient descent. GradientTape records operations so TensorFlow can calculate gradients; the loop updates W and b to reduce mean squared error.
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

# ============== CELL 5 ==============
# Cell purpose: Imports PyTorch.
import torch

# ============== CELL 6 ==============
# Cell purpose: Creates a 2×2 PyTorch tensor and displays its values, shape, and data type.
a = torch.tensor([[1, 2], [3, 4]])

print("Tensor:\n", a)
print("Shape:", a.shape)
print("Data Type:", a.dtype)

# ============== CELL 7 ==============
# Cell purpose: Creates two 1-D tensors and demonstrates element-wise arithmetic.
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print("Addition:", a+b)
print("Subtraction:", a-b)
print("Multiplication:", a*b)

# ============== CELL 8 ==============
# Cell purpose: Creates tensors filled with ones, zeros, and random values.
print("Ones:\n", torch.ones(2, 2))
print("Zeros:\n", torch.zeros(2, 2))
print("Random:\n", torch.rand(2, 2))

# ============== CELL 9 ==============
# Cell purpose: Uses autograd to differentiate y = x² + 2x + 1 at x=3. backward() computes the gradient and stores it in x.grad.
x = torch.tensor(3.0, requires_grad=True)

y = x**2 + 2*x + 1
y.backward()

print("Output:", y.item())
print("Gradient:", x.grad.item())

# ============== CELL 10 ==============
# Cell purpose: Displays a PyTorch tensor and converts it to a NumPy array.
x = torch.tensor([1, 2, 3, 4])

print("PyTorch Tensor:", x)
print("NumPy Array:", x.numpy())

# ================== THEORY: UNDERSTANDING THE TOPICS ==================
#
# 1. WHAT IS DEEP LEARNING?
# Deep learning is a part of machine learning that uses neural networks with
# multiple layers to learn patterns from examples. Instead of writing every
# rule manually, we provide examples and the model adjusts its internal
# numbers until its predictions become more accurate.
#
# 2. TENSORS: THE BASIC DATA STRUCTURE
# A tensor is a container for numbers. Its number of dimensions tells us how
# the data is arranged:
#   - Scalar (0-D): one number, e.g. 7
#   - Vector (1-D): a list, e.g. [2, 4, 6]
#   - Matrix (2-D): rows and columns, e.g. [[1, 2], [3, 4]]
#   - Higher-dimensional tensor: data arranged across more axes, such as a
#     batch of color images (image, height, width, color channel).
# The shape describes the size of each axis. For example, shape (2, 3)
# means 2 rows and 3 columns. The dtype tells the kind of values stored,
# such as integers or floating-point numbers.
#
# 3. TENSORFLOW AND PYTORCH
# TensorFlow and PyTorch are libraries used to build and train machine-learning
# models. They provide tensor math and automatic differentiation, so we do not
# need to calculate every derivative by hand. TensorFlow often uses tf.Tensor
# for values and tf.Variable for trainable values. PyTorch uses torch.Tensor;
# tensors can track gradients when requires_grad=True.
#
# 4. CONSTANTS, VARIABLES, WEIGHTS, AND BIASES
# A constant is a value that the program treats as fixed. A trainable variable
# is a value the model is allowed to change during learning.
# In the simple model y = W*x + b:
#   - x is the input,
#   - W is the weight: it controls how strongly x affects the output,
#   - b is the bias: it shifts the output up or down,
#   - y is the prediction.
# Training means finding useful values for W and b from examples.
#
# 5. LINEAR REGRESSION AS A SIMPLE MODEL
# Linear regression predicts a number using a straight-line relationship.
# For one input, its model is y_pred = W*x + b. For example, it could estimate
# a house price from its size in a simplified setting. The model starts with
# initial W and b values, makes predictions, measures its errors, and updates
# the parameters repeatedly.
#
# 6. LOSS AND MEAN SQUARED ERROR (MSE)
# A loss function measures how far predictions are from the correct answers.
# MSE = average((prediction - target)^2).
# Squaring makes negative and positive errors both count as positive and makes
# larger errors count more heavily. A smaller MSE generally means predictions
# are closer to the targets for the training examples.
#
# 7. GRADIENTS AND GRADIENT DESCENT
# A gradient tells us how the loss changes when a parameter changes. It is
# like a direction-and-slope signal: it indicates which way would increase
# the loss most rapidly. Gradient descent moves parameters in the opposite
# direction to reduce the loss:
#   parameter = parameter - learning_rate * gradient
# The learning rate controls the step size. Large steps may jump past a good
# value; very small steps can make learning slow.
#
# 8. AUTOMATIC DIFFERENTIATION
# Neural-network training needs derivatives of the loss with respect to model
# parameters. TensorFlow's GradientTape records operations inside its context
# and can calculate those derivatives. PyTorch's autograd builds a computation
# graph for tracked operations; calling backward() computes gradients and
# stores them in each relevant tensor's .grad field.
#
# 9. TRAINING LOOP: THE FULL LEARNING CYCLE
# A basic training loop repeats these steps:
#   1) Make predictions using the current parameters.
#   2) Calculate loss by comparing predictions with target values.
#   3) Calculate gradients of the loss with respect to parameters.
#   4) Update parameters using gradient descent.
#   5) Repeat for many iterations (often called steps or epochs).
# As training proceeds, the loss should generally decrease if the setup is
# suitable. The notebook's manual update demonstrates the mechanics of learning.
#
# 10. PYTORCH AUTOGRAD EXAMPLE
# If y = x^2 + 2x + 1, then dy/dx = 2x + 2. At x = 3, the derivative is 8.
# PyTorch can obtain this derivative automatically: mark x with
# requires_grad=True, compute y, and call y.backward(). The result is stored
# in x.grad. This shows how derivatives used in training are computed.
#
# 11. CONVERTING A TENSOR TO NUMPY
# NumPy is a Python library for numerical arrays. Converting a compatible
# PyTorch CPU tensor to NumPy is useful when using plotting or data-analysis
# tools that expect NumPy arrays. A tensor that is tracking gradients should
# generally be detached first (for example, tensor.detach().numpy()).
#
# 12. HOW THE NOTEBOOK TOPICS CONNECT
# Tensor operations build the model; weights and biases make predictions;
# the loss measures prediction error; automatic differentiation finds how
# parameters affect that error; gradient descent updates the parameters.
# TensorFlow GradientTape and PyTorch autograd provide the derivative step
# in two different frameworks.
#
# ================== CELL-BY-CELL EXPLANATION ==================
# Cell 1: Imports TensorFlow for tensor operations and NumPy for numerical arrays.
# Cell 2: Creates scalar TensorFlow constants and demonstrates arithmetic operations.
# Cell 3: Creates a 2×2 tensor and prints its values, shape, and data type.
# Cell 4: Trains a simple linear model y = Wx + b using gradient descent. GradientTape records operations so TensorFlow can calculate gradients; the loop updates W and b to reduce mean squared error.
# Cell 5: Imports PyTorch.
# Cell 6: Creates a 2×2 PyTorch tensor and displays its values, shape, and data type.
# Cell 7: Creates two 1-D tensors and demonstrates element-wise arithmetic.
# Cell 8: Creates tensors filled with ones, zeros, and random values.
# Cell 9: Uses autograd to differentiate y = x² + 2x + 1 at x=3. backward() computes the gradient and stores it in x.grad.
# Cell 10: Displays a PyTorch tensor and converts it to a NumPy array.
