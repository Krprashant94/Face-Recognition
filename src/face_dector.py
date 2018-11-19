# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:49:38 2018

@author: Prashant
"""

import cv2
import os

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame1 = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Streaming', frame1)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            cv2.imshow("Image", face)
            cv2.waitKey(0)
            name = input("Enter the name of person: ")
            if name != ".":
                path = "faces/"+name
                if not os.path.exists(path):
                    os.makedirs(path)
    
                _, _, files = next(os.walk(path))
                file_count = len(files)
                
                cv2.imwrite(path+'/Sample0'+str(file_count)+".jpg", face)
            
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#        if name != ".":
#            pass


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()