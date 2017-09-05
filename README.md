# Emotion-Detection-ML
Supervised Machine Learning algorithm( Logistic Regression, Support Vector Machines, Neural Network) used to detect emotion from images. The emotions that can be detected are - happy,surprise,disgust,anger and neutral.  
  
The features have been generated by first detecting 68 landmarks on the face and then calculating the angles between these landmark points and the mean of these landmark points.  
  
Cohn-Kanade(CK+) dataset has been used.  
  
Accuracy Achieved -
    Logistic Regression - 87%
    SVM - 88%
    Neural Network - 89%
  
For purpose of live demonstration, run the code detect_emotion_webcam.py and use 'Y' key on the keyboard to capture your image from the webcam

# Requirements
1. openCV 
2. DLib Library
3. numpy,matplotlib
