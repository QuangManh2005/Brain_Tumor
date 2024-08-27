import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset = []
label = []

INPUT_SIZE = 64 
#crop, resize
for image_name in no_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# First split: Train and temporary (which will be split into validation and test later)
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)  # 60% train, 40% temp


x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#regularization
model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))#softmax and sigmoid.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#learning rate, decade rate.

model.fit(x_train, y_train,
          batch_size=16,
          verbose=1,
          epochs=15, #early stoping
          validation_data=(x_test, y_test),
          shuffle=True)


model.save('BrainTumor20EpochsCategorical.h5')
