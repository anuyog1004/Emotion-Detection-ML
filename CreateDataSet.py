import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]

def ModifyImages(emotion):
    files = glob.glob("sorted_set/%s/*"%emotion)
    filenumber=0
    
    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            gray = gray[y:y+h,x:x+w]

            try:
                new_image = cv2.resize(gray,(300,350))
                cv2.imwrite("dataset/%s/%s.jpg"%(emotion,filenumber),new_image)
            except:
                pass

        filenumber+=1

for emotion in emotions:
    ModifyImages(emotion)
