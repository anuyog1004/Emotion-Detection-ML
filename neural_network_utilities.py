"""
Utility functions needed for implementing neural network.
"""

import numpy as np


def sigmoid_forward(z):
    """
    Calculates the sigmoid activation function.
    Arguments:
        z -- Input on which sigmoid function is calculated. Scalar or numpy array of any size.

    Returns:
        Output of sigmoid activation on Z.
    """
    return 1 / (1 + np.exp(-z)), z


def relu_forward(z):
    """
    Calculates the Rectified Linear Unit activation function.
    Arguments:
        z -- Input on which relu function is calculated. Scalar or numpy array of any size.

    Returns:
        Output of relu activation on Z.
    """
    return np.maximum(0, z), z


def sigmoid_backward(dA, activation_cache):
    """
    Implements derivative wrt to sigmoid activation function, needed during backpropagation algorithm.
    Arguments:
        dA -- Gradient of the cost with respect to the activation of layer l
        activation_cache -- Z

    Returns
        derivative
    """
    Z = activation_cache
    a = 1 / (1 + np.exp(-Z))
    return dA * a * (1 - a)


def relu_backward(dA, activation_cache):
    """
    Implements derivative wrt to Rectified Linear Unit activation function, needed during backpropagation algorithm.
    Arguments:
        dA -- Gradient of the cost with respect to the activation of layer l
        activation_cache -- Z

    Returns
        derivative
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
