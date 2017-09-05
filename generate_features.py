"""
This module generates the features from the images. The features have been generated by first detecting 68 landmarks on
the face and then calculating the angles between these landmark points and the mean of these landmark points.
"""
import numpy as np
import cv2
import dlib
import glob
import math


def build_features(files, label, training_file, labels_file):
    """
    Generates the features and the corresponding labels.
    Arguments:
        files: Files containing the images from which features will be calculated.
        label: Label number for the emotion.
        training_file: File to which features will be written.
        labels_file: File to which labels will be written.
    """
    for file in files:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        xlist = []
        ylist = []

        detections = detector(gray, 1)

        for k, d in enumerate(detections):
            shape = predictor(gray, d)
            for i in xrange(1, 68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)

            if xlist[26] == xlist[29]:
                anglenose = 0
            else:
                anglenose = int(
                    math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * (180 / math.pi))

            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90

            for i in xrange(0, len(xlist)):
                if xlist[i] == xmean:
                    angle = 90 - anglenose
                else:
                    angle = int(math.atan(
                        (ylist[i] - ymean) / (xlist[i] - xmean)) * (180 / math.pi)) - anglenose
                angle = str(angle)
                training_file.write(angle + " ")

            training_file.write("\n")
            labels_file.write(label + "\n")


def build_data(emotion, label):
    """
    Build the features for training data and test data.
    Arguments:
        emotion: Emotion for which features need to be generated.
        label: Label number for the emotion.
    """
    files = glob.glob('dataset/%s/*' % emotion)
    np.random.shuffle(files)
    training_data = []
    cross_validation_data = []
    training_data_length = int(0.8 * len(files))
    for i in xrange(0, training_data_length):
        training_data.append(files[i])
    for i in xrange(training_data_length + 1, len(files)):
        cross_validation_data.append(files[i])

    build_features(files=training_data, label=str(label),
                   training_file=training_features, labels_file=training_labels)
    build_features(files=cross_validation_data, label=str(label),
                   training_file=validation_features, labels_file=validation_labels)


if __name__ == '__main__':

    # Only 5 emotions have been used, since there were very less examples for
    # the remaining emotions.
    emotions = ['disgust', 'surprise', 'neutral', 'happy', 'anger']
    np.random.shuffle(emotions)

    # Ordering for the emotions.
    labels = dict()
    labels['disgust'] = 1
    labels['surprise'] = 2
    labels['neutral'] = 3
    labels['happy'] = 4
    labels['anger'] = 5

    with open('TrainingFeatures.txt', 'w') as training_features, open('ValidationFeatures.txt', 'w') as validation_features, open('TrainingLabels.txt', 'w') as training_labels, open('ValidationLabels.txt', 'w') as validation_labels:
        # Used for detecting landmarks on the face.
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        # Build training and testing data for each emotion.
        for emotion in emotions:
            build_data(emotion=emotion, label=labels[emotion])
