import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize
from sklearn import svm
import os


X = np.loadtxt('TrainingFeatures.txt',dtype=float)
y = np.loadtxt('TrainingLabels.txt')
Xcv = np.loadtxt('ValidationFeatures.txt',dtype=float)
ycv = np.loadtxt('ValidationLabels.txt')
y = np.array([y]).T
ycv = np.array([ycv]).T


linear_svm = svm.SVC(C=1,kernel='linear')
linear_svm.fit(X,y.flatten())
sc = linear_svm.score(Xcv,ycv)
print sc*100

