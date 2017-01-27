import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import glob
import os
import math

X1 = open('TrainingFeatures.txt','w')
X2 = open('ValidationFeatures.txt','w')
Y1 = open('TrainingLabels.txt','w')
Y2 = open('ValidationLabels.txt','w')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def BuildTrainingData(files,label):
	for myfile in files:
		img = cv2.imread(myfile)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		# gray = clahe.apply(gray)

		xlist = []
		ylist = []

		detections = detector(gray,1)

		for k,d in enumerate(detections):
			shape = predictor(gray,d)
			for i in range(1,68):
				xlist.append( float(shape.part(i).x) )
				ylist.append( float(shape.part(i).y) )

			xmean = np.mean(xlist)
			ymean = np.mean(ylist)

			if xlist[26] == xlist[29]:
				anglenose = 0
			else:
				anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*(180/math.pi))	
	

			if anglenose<0:
				anglenose += 90
			else:
				anglenose -= 90

			for i in range(0,len(xlist)):
				d = math.sqrt( (xlist[i]-xmean)*(xlist[i]-xmean) + (ylist[i]-ymean)*(ylist[i]-ymean) )
				if xlist[i] == xmean:
					angle = 90 - anglenose
				else:
					angle = int( math.atan( (ylist[i]-ymean)/(xlist[i]-xmean) )*(180/math.pi) ) - anglenose
				# d=str(d)
				angle=str(angle)
				# X1.write(d + " " + angle + " ")
				X1.write(angle + " ")

			X1.write("\n")
			Y1.write(label + "\n")


def BuildValidationData(files,label):
	for myfile in files:
		img = cv2.imread(myfile)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		# gray = clahe.apply(gray)

		xlist = []
		ylist = []

		detections = detector(gray,1)

		for k,d in enumerate(detections):
			shape = predictor(gray,d)
			for i in range(1,68):
				xlist.append( float(shape.part(i).x) )
				ylist.append( float(shape.part(i).y) )

			xmean = np.mean(xlist)
			ymean = np.mean(ylist)

			if xlist[26] == xlist[29]:
				anglenose = 0
			else:
				anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*(180/math.pi))	
	

			if anglenose<0:
				anglenose += 90
			else:
				anglenose -= 90

			for i in range(0,len(xlist)):
				d = math.sqrt( (xlist[i]-xmean)*(xlist[i]-xmean) + (ylist[i]-ymean)*(ylist[i]-ymean) )
				if xlist[i] == xmean:
					angle = 90 - anglenose
				else:
					angle = int( math.atan( (ylist[i]-ymean)/(xlist[i]-xmean) )*(180/math.pi) ) - anglenose
				# d=str(d)
				angle=str(angle)
				# X2.write(d + " " + angle + " ")
				X2.write(angle + " ")

			X2.write("\n")
			Y2.write(label + "\n")


def BuildData(emotion,label):
	files = glob.glob('dataset/%s/*'%emotion)
	np.random.shuffle(files)
	training_data=[]
	cross_validation_data=[]
	tdl = int( 0.8 * len(files) )
	for i in range(0,tdl):
		training_data.append(files[i])
	for i in range(tdl+1,len(files)):
		cross_validation_data.append(files[i])

	BuildTrainingData(training_data,str(label))
	BuildValidationData(cross_validation_data,str(label))


emotions = ['disgust','surprise','neutral','happy','anger']
np.random.shuffle(emotions)

labels=dict()
labels['disgust']=1
labels['surprise']=2
labels['neutral']=3
labels['happy']=4
labels['anger']=5


for emotion in emotions:
	BuildData(emotion,labels[emotion])

X1.close()
X2.close()
Y1.close()
Y2.close()