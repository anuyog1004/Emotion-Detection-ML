"""
This module captures a photo from the webcam and then detects the emotion in it. Use 'Y' key to capture the photo.
"""
import numpy as np
import cv2
import dlib
from sklearn import svm
import math

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    video_capture = cv2.VideoCapture(0)

    ### Saving the photo  ###

    while True:
        ret, img = video_capture.read()
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.imwrite('TemporaryImage.jpg', img)
            cv2.destroyAllWindows()
            break

    ### Detecting face in the photo  ####

    img = cv2.imread("TemporaryImage.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        gray = gray[y:y + h, x:x + w]
        try:
            new_image = cv2.resize(gray, (300, 350))
            cv2.imwrite("TemporaryCrop.jpg", new_image)
        except cv2.error:
                    logging.warning(cv2.error.message)

    #### Extracting features from the photo   #####
    with open("MyFaceFeatures.txt", 'w') as my_face_data:
        img = cv2.imread("TemporaryCrop.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        xlist = []
        ylist = []

        detections = detector(gray, 1)

        for k, d in enumerate(detections):
            shape = predictor(gray, d)
            for i in range(1, 68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)

            if xlist[26] == xlist[29]:
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * (180 / math.pi))

            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90

            for i in range(0, len(xlist)):
                d = math.sqrt((xlist[i] - xmean) * (xlist[i] - xmean) + (ylist[i] - ymean) * (ylist[i] - ymean))
                if xlist[i] == xmean:
                    angle = 90 - anglenose
                else:
                    angle = int(math.atan((ylist[i] - ymean) / (xlist[i] - xmean)) * (180 / math.pi)) - anglenose
                angle = str(angle)
                my_face_data.write(angle + " ")

            my_face_data.write("\n")

    #### Training the SVM algorithm  #####

    X = np.loadtxt('TrainingFeatures.txt', dtype=float)
    Y = np.loadtxt('TrainingLabels.txt')
    my_face_data = np.loadtxt('MyFaceFeatures.txt')

    linear_svm = svm.SVC(C=1, kernel='linear', probability=True)
    linear_svm.fit(X, Y.flatten())
    val = linear_svm.predict(my_face_data)

    ### Displaying prediction  ####

    if val == 1:
        print "Detected Emotion : ", "Disgust"
    if val == 2:
        print "Detected Emotion : ", "Surprise"
    if val == 3:
        print "Detected Emotion : ", "Neutral"
    if val == 4:
        print "Detected Emotion : ", "Happy"
    if val == 5:
        print "Detected Emotion : ", "Anger"
