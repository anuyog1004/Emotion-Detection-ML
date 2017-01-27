import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import os
from sklearn import svm
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

### Saving the photo  ###

while True:
	ret,img = cap.read()
	cv2.imshow('image',img)
	if cv2.waitKey(1) & 0xFF == ord('y'):
		cv2.imwrite('TemporaryImage.jpg',img)
		cv2.destroyAllWindows()
		break

### Detecting face in the photo  ####

img = cv2.imread("TemporaryImage.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
	gray = gray[y:y+h,x:x+w]
	try:
		new_image = cv2.resize(gray,(300,350))
		cv2.imwrite("TemporaryCrop.jpg",new_image)
	except:
		pass


#### Extracting features from the photo   #####

X1 = open("MyFaceFeatures.txt",'w')

img = cv2.imread("TemporaryCrop.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

xlist=[]
ylist=[]

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
		angle=str(angle)
		X1.write(angle + " ")

	X1.write("\n")

X1.close()


#### Training the SVM algorithm  #####

X = np.loadtxt('TrainingFeatures.txt',dtype=float)
y = np.loadtxt('TrainingLabels.txt')
Xcv = np.loadtxt('MyFaceFeatures.txt')

linear_svm = svm.SVC(C=1,kernel='linear',probability=True)
linear_svm.fit(X,y.flatten())
val = linear_svm.predict(Xcv)

### Displaying prediction  ####

if val==1:
	print "Detected Emotion : ", "Disgust"
if val==2:
	print "Detected Emotion : ", "Surprise"
if val==3:
	print "Detected Emotion : ", "Neutral"
if val==4:
	print "Detected Emotion : ", "Happy"
if val==5:
	print "Detected Emotion : ", "Anger"

