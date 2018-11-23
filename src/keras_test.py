# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:49:38 2018

@author: Prashant
"""

import cv2
import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

clas = ["Pamela", "Prashant"]
batchsize = 10

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

video_capture = cv2.VideoCapture(0)

   
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

model.load_weights("model_wieghts.h5")


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
        
        face = frame[y:y+h, x:x+w]
        
        save_file = "tmp/tmp.jpg"
        
        cv2.imwrite(save_file, face)

        image = cv2.imread(save_file)
        resized_image = cv2.resize(image, (64, 64))
        cv2.imwrite(save_file, resized_image)
#
        print(model.predict_classes(np.array([resized_image])))
        cv2.imshow('Live Streaming', resized_image)

    # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame

            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()