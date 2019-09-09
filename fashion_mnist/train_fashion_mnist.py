import os
import subprocess
import numpy as np
from absl import logging as alog
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

alog.set_verbosity(alog.ERROR)
print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

model = keras.Sequential([keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3,
                                              strides=2, activation='relu', name='Conv1'),
                          keras.layers.Flatten(),
                          keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
])
model.summary()

# Train Model
testing = False
epochs = 5
#tf.train.AdamOptimizer()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))



# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key

export_path = './fmnist_model'
print('export_path = {}\n'.format(export_path))
if os.path.isdir(export_path):
  print('\nAlready saved a model, cleaning up\n')

tf.compat.v1.saved_model.simple_save(
    tf.compat.v1.keras.backend.get_session(),
    export_path,
    inputs={'input_image': model.input},
    outputs={t.name:t for t in model.outputs})

print('\nSaved model:')
