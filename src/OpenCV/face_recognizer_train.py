# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:21:15 2018

@author: Prashant
"""

import cv2, os
import numpy as np
from PIL import Image

path = "train"

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml");

lables = os.listdir(path)
faceSamples=[]
ids = []
i = 0
for lable in lables:
    folder = path + os.sep + lable
    imgs = os.listdir(folder)
    i+=1
    for img in imgs:
        imagePath = folder + os.sep + img
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        
        faceSamples.append(img_numpy)
        ids.append(i)

ids = np.array(ids)
print(faceSamples, ids)
recognizer.train(faceSamples, ids)

recognizer.save('trainer.yml')