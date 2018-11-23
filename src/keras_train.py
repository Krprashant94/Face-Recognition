# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:44:00 2018

@author: Prashant
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

   
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

clas = [ "Pamela", "Prashant"]
batchsize = 10

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'C:/Users/Prashant/Documents/GitHub/Face-Recognition/src/train',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=batchsize,
        classes=clas)  # since we use binary_crossentropy loss, we need binary labels

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)
# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        'C:/Users/Prashant/Documents/GitHub/Face-Recognition/src/valid',
        target_size=(64, 64),
        batch_size=batchsize,
        classes=clas)

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=800)
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')