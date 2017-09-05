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

if __name__ == '__main__':

    # Ordering defined in the readme downloaded form CK+ dataset.
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness",
                "surprise"]

    # List for all file names in the folder "source_emotion".
    source_emotion_folders = glob.glob("source_emotion/*")

    # Copy all images from "source_images" to "sorted_set".
    for folder in source_emotion_folders:
        folder_number = "%s" % folder[-4:]
        for file_names in glob.glob("%s/*" % folder):
            for file_name in glob.glob("%s/*" % file_names):
                current_file = file_name[20:-30]
                emotion_label_file = open(file_name, 'r')

                with open(file_name, 'r') as emotion_label_file:
                    emotion = int(float(emotion_label_file.readline()))

                    sourcefile_emotion = glob.glob(
                        "source_images/%s/%s/*" % (folder_number, current_file))[-1]
                    sourcefile_neutral = glob.glob(
                        "source_images/%s/%s/*" % (folder_number, current_file))[0]

                    destination_neutral = "sorted_set/neutral/%s" % sourcefile_neutral[
                        25:]
                    destination_emotion = "sorted_set/%s/%s" % (
                        emotions[emotion], sourcefile_emotion[25:])

                    copyfile(sourcefile_neutral, destination_neutral)
                    copyfile(sourcefile_emotion, destination_emotion)

    # Used for cropping out the face from the image.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Crop face from each image in "sorted_set" and copy to "dataset".
    for emotion in emotions:
        file_names = glob.glob("sorted_set/%s/*" % emotion)
        file_number = 0

        for file_name in file_names:
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                gray = gray[y:y + h, x:x + w]

                try:
                    new_image = cv2.resize(gray, (300, 350))
                    cv2.imwrite("dataset/%s/%s.jpg" %
                                (emotion, file_number), new_image)
                except cv2.error:
                    logging.warning(cv2.error.message)

            file_number += 1
