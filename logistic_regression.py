"""
This module implements the logistic regression algorithm using sigmoidal activation function.
Average Accuracy Achieved -
    Training Data - 88%
    Testing Data - 87%
"""
import numpy as np
from neural_network_utilities import sigmoid_forward


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Arguments:
            dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
            w -- initialized vector of shape (dim, 1)
            b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient.

    Arguments:
            w -- weights
            b -- bias
            X -- data
            Y -- true "label" vector

    Return:
            cost -- cross entropy cost
            dw -- gradient of the loss with respect to w
            db -- gradient of the loss with respect to b
    """

    m = X.shape[1]

    A = sigmoid_forward(np.dot(w.T, X) + b)[0]
    cost = -  (np.sum(np.multiply(np.log(A), Y) +
                      np.multiply(np.log(1 - A), 1 - Y))) / m
    cost = np.squeeze(cost)

    dZ = A - Y

    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate):
    """
    This function optimizes w and b by running the gradient descent algorithm

    Arguments:
            w -- weights
            b -- bias
            X -- data
            Y -- true "label" vector
            num_iterations -- number of iterations of the optimization loop
            learning_rate -- learning rate of the gradient descent update rule

    Returns:
            params -- dictionary containing the weights w and bias b
            grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
            costs -- list of all the costs computed during the optimization
    """

    costs = []

    for i in xrange(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

if __name__ == '__main__':
    # Load the data
    # Shape (dimension,number_of_examples)
    X = np.loadtxt("TrainingFeatures.txt", dtype=float).T
    # Shape (dimension,number_of_examples)
    X_test = np.loadtxt("ValidationFeatures.txt", dtype=float).T
    temp_Y = np.array([np.loadtxt("TrainingLabels.txt", dtype=int)])
    temp_Y_test = np.array([np.loadtxt("ValidationLabels.txt", dtype=int)])
    Y = np.zeros((5, temp_Y.shape[1]))
    Y_test = np.zeros((5, temp_Y_test.shape[1]))

    # Convert Y to (number_of_classes,number_of_examples)
    for i in xrange(Y.shape[1]):
        Y[temp_Y[0][i] - 1][i] = 1

    # Convert Y_test to (number_of_classes,number_of_examples)
    for i in xrange(Y_test.shape[1]):
        Y_test[temp_Y_test[0][i] - 1][i] = 1

    # Normalize the input data.
    mean = np.mean(X, axis=1).reshape((X.shape[0], 1))
    var = np.var(X, axis=1).reshape((X.shape[0], 1))
    X = (X - mean) / np.sqrt(var)
    X_test = (X_test - mean) / np.sqrt(var)

    learning_rate = 0.5
    num_iterations = 1000

    w, b = initialize_with_zeros(X.shape[0])
    parameters, grads, costs = optimize(
        w, b, X, Y, num_iterations, learning_rate)
