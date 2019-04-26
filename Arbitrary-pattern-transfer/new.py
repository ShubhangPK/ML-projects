import numpy as np

# Datasets
X = np.array([[1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0],
              [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0],
              [0, 0, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0],
              [1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
y = np.array([[0], [0], [0], [1], [1], [1], [0], [1], [1], [0], [1], [0], [1], [1], [0], [0]])

X_train = X[:8, :]
X_val = X[8:12, :]
X_test = X[12:16, :]

y_train = y[:8]
y_val = y[8:12]
y_test = y[12:16]

# Parameters
W = np.random.randn(4, 2)
b = np.zeros(2)

# Forward pass function
linear_cache = {}
def linear(input):
    output = np.matmul(input, W) + b
    linear_cache["input"] = input
    return output

softmax_cache = {}
def softmax_cross_entropy(input, y):
    batch_size = input.shape[1]
    indeces = np.arange(batch_size)

    exp = np.exp(input)
    norm = (exp.T / np.sum(exp, axis=1)).T
    softmax_cache["norm"], softmax_cache["y"], softmax_cache["indeces"] = norm, y, indeces

    losses = -np.log(norm[indeces, y])
    return np.sum(losses)/batch_size

# Backward pass functions
def softmax_cross_entropy_backward():
    norm, y, indeces = softmax_cache["norm"], softmax_cache["y"], softmax_cache["indeces"]
    dloss = norm
    dloss[indeces, y] -= 1
    return dloss

def linear_backward(dout):
    input = linear_cache["input"]
    dW = np.matmul(input.T, dout)
    db = np.sum(dout, axis=0)
    return dW, db

def eval_accuracy(output, target):
    pred = np.argmax(output, axis=1)
    target = np.reshape(target, (target.shape[0]))
    correct = np.sum(pred == target)
    accuracy = correct / pred.shape[0] * 100
    return accuracy

# Training regime
for i in range(4000):
    indeces = np.random.choice(X_train.shape[0], 4)
    batch = X_train[indeces, :]
    target = y_train[indeces]

    # Forward Pass
    linear_output = linear(batch)
    loss = softmax_cross_entropy(linear_output, target)

    # Backward Pass
    dloss = softmax_cross_entropy_backward()
    dW, db = linear_backward(dloss)

    # Weight updates
    W -= 1e-2 * dW
    b -= 1e-2 * db

    # Evaluation
    if (i+1) % 100 == 0:
        accuracy = eval_accuracy(linear_output, target)
        print ("Training Accuracy: %f" % accuracy)

    if (i+1) % 500 == 0:
        accuracy = eval_accuracy(linear(X_val), y_val)
        print("Validation Accuracy: %f" % accuracy)

# Test evaluation
accuracy = eval_accuracy(linear(X_test), y_test)
print("Test Accuracy: %f" % accuracy)