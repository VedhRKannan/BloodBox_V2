from PIL import Image
from itertools import product
import os
import cv2
import numpy as np
import tensorflow as tf
# import RPi.GPIO as GPIO
import time

# GPIO.setmode(GPIO.BCM)
# GPIO.setwarnings(False)
# GPIO.setup(18,GPIO.OUT)


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h-h % d, d), range(0, w-w % d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


tile('M2.jpeg', '.', './tiled', 300)

directory = './tiled'
model_path = 'results/maleria_model/'
model = tf.keras.models.load_model(model_path)
labels = {0: 'Uninfected', 1: 'Parasitized'}
img_width, img_height = 28, 28
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

print('good')

total = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print('starting', filename)
        img_path = f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_width, img_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[np.newaxis, ...]
        image = image / 255
        prediction = probability_model.predict(image)
        prediction = np.squeeze(prediction)
        print(prediction)
        if prediction[1] >= 0.50:
            print('infected')
            total += 1
        prediction = np.argmax(prediction)
        output = labels[prediction]
print(total)

# for i in total:
#     GPIO.output(18,GPIO.HIGH)
#     time.sleep(1)
#     print("LED off")
#     GPIO.output(18,GPIO.LOW)


# total is the amount of parasites that the modal predicts to be infected.
