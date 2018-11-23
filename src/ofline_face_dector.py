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

path = "./test"
files = os.listdir(path)
for file in files:
    image = path+"/"+file
    img = cv2.imread(image)
    
    gray = cv2.imread(image, cv2.COLOR_BGR2GRAY)

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
    print(len(faces))
    for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            cv2.imshow("Image", face)
            cv2.waitKey(0)
            name = input("Enter the name of person: ")
            if name != ".":
                path = "faces/"+name
                if not os.path.exists(path):
                    os.makedirs(path)
    
                _, _, files = next(os.walk(path))
                file_count = len(files)
                save_file = path+'/Sample0'+str(file_count)+".jpg"
                cv2.imwrite(save_file, face)
                
                image = cv2.imread(save_file)
                resized_image = cv2.resize(image, (64, 64))
                cv2.imwrite(save_file, resized_image)

            cv2.destroyAllWindows()