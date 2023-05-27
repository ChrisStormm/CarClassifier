import numpy as np
import scipy.io as sio 
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
training_images = datagen.flow_from_directory(
        'cars_train/cars_train',
        target_size=(100, 100),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')
test_images = datagen.flow_from_directory(
        'cars_test/cars_test',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')

def get_labels():
    annos = sio.loadmat('../input/cars_annos.mat')
    _, total_size = annos["annotations"].shape
    print("total sample size is ", total_size)
    labels = np.zeros((total_size, 5))
    for i in range(total_size):
        path = annos["annotations"][:,i][0][0][0].split(".")
        id = int(path[0][8:]) - 1
        for j in range(5):
            labels[id, j] = int(annos["annotations"][:,i][0][j + 1][0])
    return labels
labels = get_labels()

plt.imshow(training_images[0])
#print(training_labels[0])
#print(training_images[0])
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
