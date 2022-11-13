from PIL import Image
from itertools import product
import os
import cv2
import numpy as np
import tensorflow as tf

img_width, img_height = 28, 28
img_path = 'ALL1.jpg'
cv2.imread(img_path)

model_path = 'ALL_Model'

model = tf.keras.models.load_model(model_path)

labels = {0: 'ALL', 1: 'Uninfected'}


image = cv2.imread(img_path)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

image = cv2.resize(image, (img_width, img_height))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[np.newaxis, ...]
image = image / 255
prediction = probability_model.predict(image)
prediction = np.squeeze(prediction)
print(prediction)

if prediction[0] >= 0.50:
    print('infected')
else:
    print('uninfected')
