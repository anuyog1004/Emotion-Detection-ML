"""
This module implements a L layered neural network using back propagation algorithm.
Mini batch gradient descent has been used as the optimization algorithm.
Average Accuracy Achieved -
    Training Data - 80%
    Testing Data - 79%
"""
import numpy as np
import math
from neural_network_utilities import sigmoid_forward, sigmoid_backward, relu_forward, relu_backward


def initialize_parameters(layer_dims):
    """
        This function initializes the weights and bias matrices. He initialization has been used.
        Arguments:
            layer_dims -- python array (list) containing the dimensions of each layer in the network

        Returns:
            parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            bl -- bias vector of shape (layer_dims[l], 1)
        """
    parameters = {}
    L = len(layer_dims)
    for i in xrange(1, L):
        parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(
            2 / float(layer_dims[i - 1]))
        parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def random_mini_batches(X, Y, mini_batch_size):
    """
    This function creates the mini batches to be used for performing mini batch gradient descent algorithm.
    Arguments:
        X -- data, numpy array of shape (input size, number of examples).
        Y -- true "label" vector.
        mini_batch_size -- the number of examples in each mini batch( power of 2).

    Returns:
        mini_batches -- python list containing the mini batches.
    """
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((5, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in partitioning
    for k in xrange(int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def linear_forward(A, W, b):
    """
        This function takes in activation of previous layer, weight matrix of current layer, and bias matrix of current
        layer to calculate Z of current layer.

        Arguments:
            A -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
            Z -- the input of the activation function, also called pre-activation parameter
            cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
            """
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache


def linear_activation_forward(A_prev, W, b, activation):
    """
        This function takes in the activation of previous layer, weight matrix of current layer, and bias matrix of current layer to calculate the activation of the current layer

        Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
            A -- the output of the activation function for current layer, also called the post-activation value
            cache -- a python dictionary containing "linear_cache" and "activation_cache";
                     stored for computing the backward pass efficiently
            """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid_forward(Z)
    else:
        A, activation_cache = relu_forward(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def forward_propagate(X, parameters):
    """
        This function implements the forward propagation algorithm by using relu activation (L-1) times and sigmoid
        activation fot the last layer.

        Arguments:
            X -- data, numpy array of shape (input size, number of examples)
            parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"

        Returns:
            AL -- last post-activation value
            caches -- list of caches containing:
                        every cache of linear_relu_forward()
                        the cache of linear_sigmoid_forward()
            """

    caches = []
    L = len(parameters) // 2
    A_prev = X

    for i in xrange(L - 1):
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(i + 1)], parameters["b" + str(i + 1)], "relu")
        caches.append(cache)
        A_prev = A

    A, cache = linear_activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return A, caches


def compute_cost(AL, Y):
    """
        Implement the cross entropy cost function.

        Arguments:
            AL -- probability vector corresponding to the label predictions.
            Y -- true "label" vector

        Returns:
            cost -- cross-entropy cost
            """
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, linear_cache):
    """
        This function takes in dZ of current layer and calculates dW of current layer, db of current layer and dA of previous layer

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        linear_cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA = np.dot(W.T, dZ)

    return dA, dW, db


def linear_activation_backward(dA, cache, activation):
    """
        This function takes in dA of current layer and calculates dW of current layer, db of current layer and dA of previous layer.

        Arguments:
            dA -- post-activation gradient for current layer l
            cache -- tuple of values (linear_cache, activation_cache) stored for computing backward propagation efficiently
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        dZ = relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def back_propagate(AL, Y, caches):
    """
        This function implements the back propagation algorithm by using linear_activation_backward with relu activation (L-1)
        times and sigmoid activation once.

        Arguments:
            AL -- probability vector, output of the forward propagation
            Y -- true "label" vector
            caches -- list of caches of the form (linear_cache,activation_cache)

        Returns:
            grads -- A dictionary with the gradients
                     grads["dA" + str(l)] = ...
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ...
            """
    grads = {}
    m = AL.shape[1]
    L = len(caches)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")

    for l in reversed(xrange(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    This function performs the updates the parameters needed for gradient descent algorithm.
    Arguments:
        parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"
        grads -- dictionary with the gradients
        learning_rate -- needed foe updating the parameters

    Returns:
        parameters -- the updated dictionary containing parameters "W1", "b1", ..., "WL", "bL"
    """
    L = len(parameters) / 2

    for i in xrange(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - (learning_rate * grads["dW" + str(i + 1)])
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - (learning_rate * grads["db" + str(i + 1)])

    return parameters


def print_accuracy(X, Y, parameters):
    """
    This function prints the accuracy by using the learned parameters.
    Arguments:
        X -- data on which accuracy needs to be calculated
        Y -- true "label" vector
        parameters -- python dictionary containing the learned "W1", "b1", ..., "WL", "bL"

    Returns:
        accuracy
    """
    AL, caches = forward_propagate(X, parameters)
    m = X.shape[1]
    correct = 0
    for i in xrange(m):
        training = np.array([Y[:, i]]).T
        predicted = np.array([AL[:, i]]).T
        if np.argmax(training) == np.argmax(predicted):
            correct += 1

    return (correct / float(m)) * 100


if __name__ == '__main__':
    # Load the data
    X = np.loadtxt("TrainingFeatures.txt", dtype=float).T  # Shape (dimension,number_of_examples)
    X_test = np.loadtxt("ValidationFeatures.txt", dtype=float).T  # Shape (dimension,number_of_examples)
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

    # Normalise X
    for i in xrange(X.shape[1]):
        mean = np.mean(X[:, i])
        var = np.var(X[:, i])
        X[:, i] = (X[:, i] - mean) / var

    # Normalise X_test
    for i in xrange(X_test.shape[1]):
        mean = np.mean(X_test[:, i])
        var = np.var(X_test[:, i])
        X_test[:, i] = (X_test[:, i] - mean) / var

    # Initialize neural network architecture.
    layer_dims = [X.shape[0], 50, Y.shape[0]]  # Since data is very less, using a single layer gives the most accuracy.
    parameters = initialize_parameters(layer_dims)
    costs = []
    number_of_iterations = 75000
    learning_rate = 0.4

    mini_batches = random_mini_batches(X, Y, 32)

    # Mini Batch Gradient Descent Algorithm.
    for i in xrange(number_of_iterations):

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch

            AL, caches = forward_propagate(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)

            if i % 100 == 0:
                costs.append(cost)

            grads = back_propagate(AL, mini_batch_Y, caches)

            parameters = update_parameters(parameters, grads, learning_rate)

    print "Training Set Accuracy : ", print_accuracy(X, Y, parameters)
    print "Test Set Accuracy : ", print_accuracy(X_test, Y_test, parameters)
