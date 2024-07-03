#last training: 250/40000 iter, 32 batch, 0.1 alpha, 0.001 decay rate, 98.05% accurate

"""
Input Layer           Hidden Layer 1         Hidden Layer 2         Output Layer
(784 neurons)         (128 neurons)         (64 neurons)           (10 neurons)
    |                       |                      |                      |
    |                       |                      |                      |
    v                       v                      v                      v
   [ ]-------------------->[ ]------------------->[ ]------------------->[ ]
    |       (W1, b1)        |       (W2, b2)       |       (W3, b3)       |
    |                       |                      |                      |
    v                       v                      v                      v
  Activation:              Activation:            Activation:            Activation:
  Leaky ReLU               Leaky ReLU             Leaky ReLU             Softmax

"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, shift

data = pd.read_csv('data/train.csv')
data = np.array(data)
data[:, 1:] = np.where(data[:, 1:] < 1, 0, 255)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:100].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def plot_digit_distribution(Y):
    unique, counts = np.unique(Y, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color='blue')
    plt.xlabel('Digits')
    plt.ylabel('Frequency')
    plt.title('Distribution of Digits in Training Data')
    plt.xticks(unique)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(2. / 784)  # He initialization
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(2. / 128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(2. / 64)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def leaky_relu(Z, alpha=0.01):
    return np.maximum(alpha * Z, Z)

def leaky_relu_deriv(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

    
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = leaky_relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = leaky_relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * leaky_relu_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * leaky_relu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, 
                  v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, alpha, momentum):
    v_W1 = momentum * v_W1 + (1 - momentum) * dW1
    v_b1 = momentum * v_b1 + (1 - momentum) * db1
    v_W2 = momentum * v_W2 + (1 - momentum) * dW2
    v_b2 = momentum * v_b2 + (1 - momentum) * db2
    v_W3 = momentum * v_W3 + (1 - momentum) * dW3
    v_b3 = momentum * v_b3 + (1 - momentum) * db3

    W1 = W1 - alpha * v_W1
    b1 = b1 - alpha * v_b1
    W2 = W2 - alpha * v_W2
    b2 = b2 - alpha * v_b2
    W3 = W3 - alpha * v_W3
    b3 = b3 - alpha * v_b3

    return W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3

def data_augmentation(X, Y):
    augmented_X = []
    augmented_Y = []
    for i in range(X.shape[1]):
        img = X[:, i].reshape(28, 28)
        augmented_X.append(X[:, i])
        augmented_Y.append(Y[i])
        
        # Rotate
        rotated = rotate(img, angle=np.random.uniform(-15, 15), reshape=False)
        augmented_X.append(rotated.flatten())
        augmented_Y.append(Y[i])
        
        # Shift
        shifted = shift(img, [np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
        augmented_X.append(shifted.flatten())
        augmented_Y.append(Y[i])
    
    return np.array(augmented_X).T, np.array(augmented_Y)

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def init_velocity():
    v_W1 = np.zeros((128, 784))
    v_b1 = np.zeros((128, 1))
    v_W2 = np.zeros((64, 128))
    v_b2 = np.zeros((64, 1))
    v_W3 = np.zeros((10, 64))
    v_b3 = np.zeros((10, 1))
    return v_W1, v_b1, v_W2, v_b2, v_W3, v_b3


def stochastic_gradient_descent(X, Y, alpha, iterations, batch_size, decay_rate, momentum=0.9):
    final_accuracy = 0
    W1, b1, W2, b2, W3, b3 = init_params()
    v_W1, v_b1, v_W2, v_b2, v_W3, v_b3 = init_velocity()
    accuracy_list = []
    learning_rates = []
    
    X_train, X_val, Y_train, Y_val = train_test_split(X.T, Y, test_size=0.1, random_state=42)
    X_train, Y_train = data_augmentation(X_train.T, Y_train)
    X_val = X_val.T
    
    m = X_train.shape[1]
    
    for i in range(iterations):
        current_alpha = alpha * (1 / (1 + decay_rate * i))
        learning_rates.append(current_alpha)
        
        # Randomly shuffle the data
        permutation = np.random.permutation(m)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[permutation]
        
        # Mini-batch training
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]
    
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)
            W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, current_alpha, momentum)
        
        if i % 10 == 0:  # Evaluate less frequently to save time
            _, _, _, _, _, A3_val = forward_prop(W1, b1, W2, b2, W3, b3, X_val)
            predictions = get_predictions(A3_val)
            accuracy = get_accuracy(predictions, Y_val)
            print(f"Iteration: {i}, Accuracy: {accuracy}, Learning Rate: {current_alpha}")
            accuracy_list.append(accuracy)
            final_accuracy = accuracy

            with open('parameters_stoch_mom.pkl', 'wb') as f:
                pickle.dump((W1, b1, W2, b2, W3, b3), f)

    fig, ax1 = plt.subplots()

    ax1.plot(range(0, iterations, 10), accuracy_list, 'b-')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(range(iterations), learning_rates, 'r-')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params('y', colors='r')

    plt.title(f'Accuracy and Learning Rate over Iterations (initial alpha={alpha}, decay_rate={decay_rate}, iterations={iterations}, final accuracy={final_accuracy:.2f}%)')
    plt.savefig(os.path.join('Training', f'accuracy_learning_rate-{alpha}-{decay_rate}-{iterations}-{batch_size}-{momentum}-{final_accuracy:.2f}.png'))
    
    return W1, b1, W2, b2, W3, b3, final_accuracy 

iterations = 250
batch_size = 32
alpha = 0.1
decay_rate = 0.001
momentum = 0.9

#Y_train_first_500 = Y_train[:500]
#plot_digit_distribution(Y_train_first_500)

W1, b1, W2, b2, W3, b3, accuracy = stochastic_gradient_descent(X_train, Y_train, alpha, iterations, batch_size, decay_rate, momentum) 

"""
 Given:

 - W1, b1: Weights and biases for the first hidden layer.
 - W2, b2: Weights and biases for the second hidden layer.
 - W3, b3: Weights and biases for the output layer.
 - dW1, db1: Gradients of loss w.r.t. W1 and b1.
 - dW2, db2: Gradients of loss w.r.t. W2 and b2.
 - dW3, db3: Gradients of loss w.r.t. W3 and b3.
 - v_W1, v_b1: Velocity terms for W1 and b1 (used in momentum).
 - v_W2, v_b2: Velocity terms for W2 and b2 (used in momentum).
 - v_W3, v_b3: Velocity terms for W3 and b3 (used in momentum).
 - alpha: Learning rate.
 - beta: Momentum parameter (typically set to 0.9).

 Update steps:

 1. Velocity Update (Momentum Update):
    v_W1 = beta * v_W1 + (1 - beta) * dW1
    v_b1 = beta * v_b1 + (1 - beta) * db1
    v_W2 = beta * v_W2 + (1 - beta) * dW2
    v_b2 = beta * v_b2 + (1 - beta) * db2
    v_W3 = beta * v_W3 + (1 - beta) * dW3
    v_b3 = beta * v_b3 + (1 - beta) * db3

 2. Weight and Bias Update:

    W1 = W1 - alpha * v_W1
    b1 = b1 - alpha * v_b1
    W2 = W2 - alpha * v_W2
    b2 = b2 - alpha * v_b2
    W3 = W3 - alpha * v_W3
    b3 = b3 - alpha * v_b3
"""