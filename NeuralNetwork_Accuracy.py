import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize
import scipy.misc
import random
import matplotlib.cm as cm
from scipy.special import expit
import itertools

# Loading the data

X = np.loadtxt("TrainingFeatures.txt",dtype=float)
y = np.loadtxt("TrainingLabels.txt",dtype=int)
Xcv = np.loadtxt("ValidationFeatures.txt",dtype=float)
ycv = np.loadtxt("ValidationLabels.txt",dtype=int)
y = np.array([y]).T
ycv = np.array([ycv]).T
X = np.insert(X,0,1,axis=1)
Xcv = np.insert(Xcv,0,1,axis=1)


# Rolling and unrolling of parameters

def flattenParams(thetas_list):
    flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (67+1)*50 + (50+1)*5
    return np.array(combined).reshape((len(combined),1))

def reshapeParams(flattened_array):
    theta1 = flattened_array[:(67+1)*50] .reshape((50,67+1))
    theta2 = flattened_array[(67+1)*50:] .reshape((5,50+1))
    
    return [ theta1, theta2 ]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((446*(67+1),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((446,67+1))

# Forward Propagation

def ForwardPropagate(row,theta):  
	features = row
	for i in range(0,len(theta)):
		Theta = theta[i]
		z = np.dot(Theta,features.T)
		a = expit(z)
		if i == len(theta)-1:
			return a
		a = np.insert(a,0,1)
		features = a.T    

# Cost Function to minimize

def GetCost(theta,X,y,ld):       
	theta = reshapeParams(theta)
	X = reshapeX(X)
	total_cost = 0
	m = X.shape[0]
	for i in range(0,m):
		row = X[i,:]
		hs = ForwardPropagate(row,theta)
		tempy = np.zeros((5,1))
		tempy[y[i]-1]=1
		t1 = -np.dot(tempy.T,np.log(hs))
		t2 = -np.dot( (1-tempy).T, np.log(1-hs) )
		cost = t1 + t2
		total_cost += cost
			
	reg_cost = 0
	for Theta in theta:
		reg_cost += np.sum(Theta*Theta)

	reg_cost *= float(ld)/(2*m)

	return float((1./m)*total_cost) + reg_cost

def SigmoidGradient(z):   
	temp = expit(z)
	return temp*(1-temp)

def RandomInitialisation(eps):
	th1 = np.random.uniform(-eps,eps,(50,68))
	th2 = np.random.uniform(-eps,eps,(5,51))
	th = [th1,th2]
	return th

# Back Propagation

def BackProp(theta,X,y,ld):
	theta = reshapeParams(theta)
	X = reshapeX(X)
	theta1 = theta[0]
	theta2 = theta[1]
	D1 = np.zeros((50,68))
	D2 = np.zeros((5,51))
	m = X.shape[0]
	for i in range(0,m):
		row = X[i,:]
		a3 = np.array([ForwardPropagate(row,theta)]).T
		tempy = np.zeros((5,1))
		tempy[y[i]-1]=1
		d3 = a3 - tempy
		z2 = np.dot(theta1,row.T)
		z2 = np.insert(z2,0,1)
		g = SigmoidGradient(z2)
		a2 = np.array(expit(z2))
		g = np.array([g]).T
		a2 = np.array([a2]).T
		d2 = np.dot(theta2.T,d3) * g
		D2 += np.dot(d3,a2.T)
		d2 = d2[1:]
		row = np.array([row])
		D1 += np.dot(d2,row)

	D1 = D1/float(m)
	D2 = D2/float(m)
	D1[:,1:] += float(ld/m)*theta1[:,1:]
	D2[:,1:] += float(ld/m)*theta2[:,1:]
	return flattenParams([D1,D2]).flatten()

def NeuralNetwork(X,y):
	theta = RandomInitialisation(0.5)
	theta = flattenParams(theta)
	X = flattenX(X)
	result = scipy.optimize.fmin_cg(GetCost,x0=theta,fprime=BackProp,args=(X,y,1),maxiter=100000,disp=True,full_output=True)
	return result[0]

ltheta = NeuralNetwork(X,y)	
ltheta = reshapeParams(ltheta)

def NeuralAccuracy(X,y,Xcv,ycv):
	total = Xcv.shape[0]
	correct = 0
	for i in range(0,total):
		ans = ForwardPropagate(Xcv[i],ltheta)
		res = -1
		val = -1
		for j in range(0,len(ans)):
			if ans[j] > res:
				res = ans[j]
				val = j+1

		if val == ycv[i] :
			correct += 1

	print "Total : ", total
	print "Correct : ", correct
	print "Accuracy : ", float(correct)/float(total) * 100

NeuralAccuracy(X,y,Xcv,ycv)
