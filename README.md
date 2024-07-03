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
