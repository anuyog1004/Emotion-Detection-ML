"""
This module implements the SVM algorithm.
Average Accuracy Achieved -
    Training Data - 87%
    Testing Data - 88%
"""
import numpy as np
from sklearn import svm

# Load the data.
X = np.loadtxt('TrainingFeatures.txt', dtype=float).T
Y = np.loadtxt('TrainingLabels.txt')
X_test = np.loadtxt('ValidationFeatures.txt', dtype=float).T
Y_test = np.loadtxt('ValidationLabels.txt')
Y = np.array([Y]).T
Y_test = np.array([Y_test]).T

# Normalize the input data.
mean = np.mean(X, axis=1).reshape((X.shape[0], 1))
var = np.var(X, axis=1).reshape((X.shape[0], 1))
X = (X - mean) / np.sqrt(var)
X_test = (X_test - mean) / np.sqrt(var)

X = X.T
X_test = X_test.T

linear_svm = svm.SVC(C=1, kernel='linear')
linear_svm.fit(X, Y.flatten())
training_accuracy = linear_svm.score(X, Y)
testing_accuracy = linear_svm.score(X_test, Y_test)

print "Training accuracy ", training_accuracy * 100
print "Testing accuracy ", testing_accuracy * 100
