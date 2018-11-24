# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:21:15 2018

@author: Prashant
"""

import cv2
import numpy as np
import os 

database = ["0", "Pamela", "Prashant"]
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml");

recognizer.read('trainer.yml')

font = cv2.FONT_HERSHEY_PLAIN
cam = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = detector.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist 
#        if(Id == 1):
#            Id = "Nazmi {0:.2f}%".format(round(100 - confidence, 2))

        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, database[Id], (x,y-40), font, 2, (255,255,255), 1)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
