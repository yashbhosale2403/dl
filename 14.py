# Word Embeddings with a CBOW-style Model
# Converted from DL_Chit_14.ipynb into a cell-wise Python script.
# Run cells in order in Jupyter or as a Python script.

# ==================== CELL 1 ====================
# Cell purpose: Imports libraries and Keras components for text processing and the neural model.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.utils import to_categorical

# ==================== CELL 2 ====================
# Cell purpose: Defines a small example corpus of sentences.
sentences = [
    "i love machine learning",
    "machine learning is interesting",
    "i love deep learning",
    "deep learning uses neural networks",
    "neural networks learn from data",
    "machine learning uses data",
    "i enjoy learning new things",
    "deep neural networks are powerful"
]

print("Total sentences:", len(sentences))

# ==================== CELL 3 ====================
# Cell purpose: Tokenizes the corpus and converts words into integer sequences.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(sentences)

print("Vocabulary:", word_index)
print("Vocabulary Size:", vocab_size)

# ==================== CELL 4 ====================
# Cell purpose: Creates context-target training pairs using a sliding window.
window_size = 2
data = []

for seq in sequences:
    for i in range(window_size, len(seq) - window_size):
        context = seq[i-window_size:i] + seq[i+1:i+window_size+1]
        target = seq[i]
        data.append((context, target))

X = np.array([x[0] for x in data])
y = np.array([x[1] for x in data])

print("Context Shape:", X.shape)
print("Target Shape:", y.shape)

# ==================== CELL 5 ====================
# Cell purpose: Prints example context words and their center-word targets.
index_word = {v: k for k, v in word_index.items()}

for context, target in data[:10]:
    words = [index_word[i] for i in context]
    print("Context:", words, "-> Target:", index_word[target])

# ==================== CELL 6 ====================
# Cell purpose: Builds an embedding model that predicts a center word from its context.
embedding_dim = 50

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=window_size*2),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(vocab_size, activation='softmax')
])

model.summary()

# ==================== CELL 7 ====================
# Cell purpose: Configures the model's optimizer, loss, and accuracy metric.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== CELL 8 ====================
# Cell purpose: Trains the model on the generated context-target pairs.
history = model.fit(
    X, y,
    epochs=200,
    batch_size=8,
    verbose=1
)

# ==================== CELL 9 ====================
# Cell purpose: Plots training loss and accuracy across epochs.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epochs')

plt.tight_layout()
plt.show()

# ==================== CELL 10 ====================
# Cell purpose: Extracts learned word embeddings and ranks words by cosine similarity.
embeddings = model.layers[0].get_weights()[0]

word = "learning"
word_vector = embeddings[word_index[word]]

similarity = np.dot(embeddings, word_vector) / (
    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vector) + 1e-10
)

similar_words = np.argsort(similarity)[::-1][1:6]

print("Words similar to:", word)

for idx in similar_words:
    print(index_word.get(idx, "Unknown"), similarity[idx])

# ==================== THEORY / VIVA NOTES ====================
# 1. Tokenization:
#    Splits text into words and assigns each word an integer ID.
#
# 2. Vocabulary:
#    The set of unique words in the corpus. Keras reserves index 0, so
#    vocab_size is the number of unique words plus one.
#
# 3. Context window:
#    Nearby words are used to predict a target word. Here window_size=2
#    means two words on the left and two words on the right.
#
# 4. CBOW (Continuous Bag of Words):
#    A Word2Vec approach that predicts a center word from surrounding words.
#    The context word order is not explicitly modeled in this example.
#
# 5. Embedding layer:
#    Maps each word ID to a trainable dense vector of embedding_dim values.
#    These vectors are learned during training.
#
# 6. Mean pooling:
#    tf.reduce_mean averages the context word vectors into one representation.
#
# 7. Softmax output:
#    Produces a probability distribution over the vocabulary for the target.
#
# 8. Loss and optimizer:
#    Sparse categorical cross-entropy compares predicted probabilities with
#    integer target IDs. Adam updates model weights to reduce the loss.
#
# 9. Cosine similarity:
#    Measures the angle between embedding vectors. Values closer to 1 indicate
#    vectors pointing in similar directions, often suggesting related words.
#
# 10. Limitations:
#     This is a tiny demonstration corpus, so learned embeddings are unstable
#     and should not be treated as reliable semantic representations.
