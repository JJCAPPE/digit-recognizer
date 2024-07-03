import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

data = pd.read_csv('data/train.csv')
data = np.array(data)
data[:, 1:] = np.where(data[:, 1:] < 1, 0, 255)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    accuracy_list = []
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            accuracy_list.append(accuracy)
            final_accuracy = accuracy

        with open('parameters.pkl', 'wb') as f:
            pickle.dump((W1, b1, W2, b2), f)
    
    return accuracy_list, final_accuracy

lowerBound = 0.54
upperBound = 0.58
increments = 0.02
iterations = 2500

alphas = np.arange(lowerBound, upperBound, increments)

all_accuracies = []
final_accuracies = []

for alpha in alphas:
    print(f"Running gradient descent for alpha = {alpha}...")
    accuracy_list, final_accuracy = gradient_descent(X_train, Y_train, alpha, iterations)
    all_accuracies.append(accuracy_list)
    final_accuracies.append(final_accuracy)
    
    plt.plot(range(0, iterations, 10), accuracy_list)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over iterations (alpha={alpha}, iterations=500, final accuracy={final_accuracy:.2f}%)')
    plt.savefig(os.path.join('Plots', f'accuracy-{alpha}-500-{final_accuracy:.2f}.png'))
    plt.close()

plt.figure()
plt.plot(alphas, final_accuracies, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Final Accuracy')
plt.title('Final Accuracy vs. Alpha')
plt.grid(True)
plt.savefig(os.path.join('Plots', 'alpha_vs_accuracy.png'))
plt.show()