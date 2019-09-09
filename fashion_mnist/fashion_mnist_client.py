import os
import random
import requests
import json
import subprocess
import cv2
import numpy as np
from absl import logging as alog
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

SERVER_URL = 'http://localhost:8501/v1/models/fmnist_model:predict'
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

alog.set_verbosity(alog.ERROR)
print(tf.__version__)


def show(test_image, title):
    # plt.figure()
    # plt.imshow(test_image.reshape(28,28))
    # plt.axis('off')
    # plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    resized = cv2.resize(test_image, (128, 128))
    cv2.imshow(title, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


    # Show a random image:
    rando = random.randint(0,len(test_images)-1)
    show(test_images[rando], CLASS_NAMES[test_labels[rando]])

    # Run inference through fashion mnist model being served by tensorflow serving docker container.abs(x)
    data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
    headers = {"content-type": "application/json"}
    print("Before request!!!!")
    json_response = requests.post(SERVER_URL, data=data, headers=headers)
    print("json_response:", json_response)
    predictions = json.loads(json_response.text)['predictions']

if __name__ == '__main__':
    main()
