import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy import optimize

X = np.loadtxt('TrainingFeatures.txt',dtype=float)
y = np.loadtxt('TrainingLabels.txt')
Xcv = np.loadtxt('ValidationFeatures.txt',dtype=float)
ycv = np.loadtxt('ValidationLabels.txt')
y = np.array([y]).T
ycv = np.array([ycv]).T


def h(myX,myTheta):
	return expit( np.dot(myX,myTheta) )


def GetCost(myTheta,myX,myy,ld):
	m = myX.shape[0]
	term1 = np.dot( myy.T , np.log( h(myX,myTheta) ) )
	term2 = np.dot( 1-(myy).T , np.log(1-h(myX,myTheta)) )
	regterm = float(ld)/2 * np.sum(np.dot(myTheta[1:].T,myTheta[1:]))
	cost = term1 + term2
	return float(-1./m)*(cost + regterm)


def OptimizeTheta(ltheta,myX,myy,ld):

	i=1
	while i<=5:
		theta = np.zeros((myX.shape[1],1))
		tempy = np.zeros((y.shape[0],y.shape[1]))
		for j in range(0,tempy.shape[0]):
			if y[j] == i :
				tempy[j] = 1

		res = optimize.fmin(GetCost,x0=theta,args=(myX,tempy,ld),maxiter=100000,full_output=True)
		ans = res[0]

		for k in range(0,ans.shape[0]):
			ltheta[k,i-1] = ans[k]

		i+=1


ltheta = np.zeros((X.shape[1],5))
OptimizeTheta(ltheta,X,y,0)

def GetAccuracy(ltheta,myX,myy):
	m = myX.shape[0]
	correct = 0
	for i in range(0,m):
		val = -1
		res = -1
		for j in range(0,5):
			myTheta = ltheta[:,j]
			myTheta = np.array([myTheta]).T
			tempX = np.array([myX[i]])
			ans = h(tempX,myTheta)
			if ans > res :
				res = ans
				val = j+1

		if val == myy[i]:
			correct += 1

        print "Correctly Classified", correct
        print "Total", m
        print "Accuracy", float(correct)/float(m) * 100


GetAccuracy(ltheta,Xcv,ycv)
