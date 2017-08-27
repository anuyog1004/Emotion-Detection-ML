"""
1. This module copies the images from the folder "source_images"( extracted from CK + dataset )
   into a folder "sorted_set" which contains images sorted by the emotion.
   e.g. The folder angry in sorted_set contains all angry images, happy contains all happy images and so on.
2. Each image in the sorted_set folder is then cropped to (300,350) image using OpenCV haarcascade and copied to a new
   folder  "dataset". Each image in the folder "datset" contains only the face displaying the emotion.
   This is done to remove noise.
"""
import glob
import cv2
import logging
from shutil import copyfile

# Ordering defined in the readme downloaded form CK+ dataset.
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness",
            "surprise"]

# List for all file names in the folder "source_emotion".
participants = glob.glob("source_emotion/*")

# Copy all images from "source_images" to "sorted_set".
for x in participants:
    part = "%s" % x[-4:]
    for sessions in glob.glob("%s/*" % x):
        for files in glob.glob("%s/*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(float(file.readline()))

            sourcefile_emotion = glob.glob("source_images/%s/%s/*" % (part, current_session))[-1]
            sourcefile_neutral = glob.glob("source_images/%s/%s/*" % (part, current_session))[0]

            dest_neut = "sorted_set/neutral/%s" % sourcefile_neutral[25:]
            dest_emot = "sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[25:])

            copyfile(sourcefile_neutral, dest_neut)
            copyfile(sourcefile_emotion, dest_emot)

# Used for cropping out the face from the image.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Crop face from each image in "sorted_set" and copy to "dataset".
for emotion in emotions:
    files = glob.glob("sorted_set/%s/*" % emotion)
    file_number = 0

    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            gray = gray[y:y + h, x:x + w]

            try:
                new_image = cv2.resize(gray, (300, 350))
                cv2.imwrite("dataset/%s/%s.jpg" % (emotion, file_number), new_image)
            except cv2.error:
                logging.warning(cv2.error.message)

        file_number += 1
