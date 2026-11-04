# ==================== CELL 1 ====================
# Cell purpose: Imports NumPy and Matplotlib and defines the sigmoid activation and its derivative.
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# ==================== CELL 2 ====================
# Cell purpose: Defines a simple recurrent neural network and its forward and backward passes.
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.lr = learning_rate

        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, output_size) * 0.01

        self.b_h = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, output_size))

    def forward(self, inputs):
        self.hidden_states, self.outputs = {}, {}
        h = np.zeros((1, self.hidden_size))

        for t, x in enumerate(inputs):
            h = sigmoid(x.reshape(1, -1) @ self.W_xh + h @ self.W_hh + self.b_h)
            self.hidden_states[t] = h
            self.outputs[t] = h @ self.W_ho + self.b_o

        return self.outputs[len(inputs)-1]

    def backward(self, inputs, targets):
        dW_xh, dW_hh, dW_ho = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_ho)
        db_h, db_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        dh_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(len(inputs))):
            x = inputs[t].reshape(1, -1)
            h = self.hidden_states[t]
            h_prev = self.hidden_states[t-1] if t > 0 else np.zeros_like(h)

            dy = self.outputs[t] - targets[t].reshape(1, -1)
            dW_ho += h.T @ dy
            db_o += dy

            dh_raw = (dy @ self.W_ho.T + dh_next) * sigmoid_derivative(h)

            dW_xh += x.T @ dh_raw
            dW_hh += h_prev.T @ dh_raw
            db_h += dh_raw

            dh_next = dh_raw @ self.W_hh.T

        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.W_ho -= self.lr * dW_ho
        self.b_h -= self.lr * db_h
        self.b_o -= self.lr * db_o


# ==================== CELL 3 ====================
# Cell purpose: Creates a small input/target sequence, initializes the RNN, and trains it for 1000 epochs.
# This practical uses generated data, so no external dataset or download is needed.
# Set the seed so the random initialization is repeatable.
np.random.seed(42)

input_sequence = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
target_sequence = np.array([[0.2], [0.4], [0.6], [0.8], [1.0]])

# Create RNN
rnn = SimpleRNN(1, 5, 1, learning_rate=0.05)

print("Training Simple RNN...")

# Training
for epoch in range(1000):
    predicted = rnn.forward(input_sequence)

    predicted_sequence = np.array([
        rnn.outputs[t] for t in range(len(input_sequence))
    ])

    loss = np.mean((predicted_sequence - target_sequence) ** 2)

    rnn.backward(input_sequence, target_sequence)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


# ==================== CELL 4 ====================
# Cell purpose: Runs the trained RNN on the input sequence and plots its output at each time step.
# Generate final predictions
rnn.forward(input_sequence)

predictions = [
    rnn.outputs[t] for t in range(len(input_sequence))
]

values = [output[0][0] for output in predictions]

# Plot RNN Output
plt.plot(values, marker="o")

plt.title("RNN Output")
plt.xlabel("Time Steps")
plt.ylabel("Output")

plt.grid()
plt.show()


# ================== THEORY: SIMPLE RECURRENT NEURAL NETWORK ==================
#
# 1. RECURRENT NEURAL NETWORK (RNN)
# An RNN is a neural network designed for sequential data. It processes one
# time step at a time and carries information forward through a hidden state.
# Examples of sequential data include text, speech, and time-series values.
#
# 2. INPUT AND TARGET SEQUENCES
# The input_sequence contains five values: 0.1, 0.2, 0.3, 0.4, and 0.5.
# The target_sequence contains the desired outputs: 0.2, 0.4, 0.6, 0.8, and 1.0.
# These are small generated arrays, so this practical does not require a dataset.
#
# 3. SIGMOID ACTIVATION
# Sigmoid maps a value to the range (0, 1): sigmoid(x) = 1 / (1 + exp(-x)).
# It introduces nonlinearity in the hidden state. Its derivative can be computed
# from its output as h * (1 - h).
#
# 4. RNN WEIGHTS
# W_xh connects the current input to the hidden layer. W_hh connects the previous
# hidden state to the current hidden state. W_ho maps the hidden state to output.
# b_h and b_o are the hidden and output biases.
#
# 5. FORWARD PROPAGATION
# At each time step, the RNN computes a hidden state from the current input and
# previous hidden state, then computes an output from that hidden state.
# The initial hidden state is all zeros.
#
# 6. HIDDEN STATE
# The hidden state acts as a compact memory of information seen so far.
# Because it is passed from one time step to the next, earlier inputs can affect
# later outputs.
#
# 7. BACKPROPAGATION THROUGH TIME (BPTT)
# Training propagates errors backward through the sequence. Gradients for shared
# weights are accumulated across time steps, then used to update the parameters.
# dh_next carries the gradient contribution from later time steps.
#
# 8. MEAN SQUARED ERROR (MSE)
# MSE is the average squared difference between predicted and target values.
# Smaller MSE means predictions are closer to the target sequence.
#
# 9. LEARNING RATE AND EPOCHS
# The learning rate controls the size of each weight update. An epoch is one
# training pass over the complete sequence. Too large a learning rate can cause
# loss to increase or training to become unstable.
#
# 10. PREDICTION AND PLOT
# After training, the RNN processes the input sequence again. The plot displays
# its output at each time step, showing how the output changes over the sequence.
#
# 11. LIMITATIONS
# This is a small educational RNN implemented with NumPy. Simple RNNs can have
# vanishing or exploding gradients on long sequences and may struggle to retain
# long-term information. LSTM and GRU networks help address these limitations.
#
# 12. WORKFLOW
# Import libraries → define activation functions → define RNN → create sequences
# → train with BPTT → generate predictions → plot outputs.
#
# ==================== CELL-BY-CELL EXPLANATION ====================
# Cell 1: Imports libraries and defines sigmoid and its derivative.
# Cell 2: Defines RNN weights, forward propagation, and BPTT weight updates.
# Cell 3: Creates toy sequences and trains the RNN for 1000 epochs.
# Cell 4: Generates the final outputs and plots them.
