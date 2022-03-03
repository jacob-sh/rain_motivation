import random
random.seed(20)

import numpy as np
np.random.seed(20)

import tensorflow as tf
tf.random.set_seed(20)

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# download and prepare the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# the seeds used for training the models
seeds = {'314', '159', '265', '358', '979', '323', '846', '264', '338', '327', '950', '288', '419', '716', '939', '937', '510', '582', '097', '494'}

# load all models in a dictionary
models = {}
for seed in seeds:
    models[seed] = keras.models.load_model('./model_' + seed)


for image, label in zip(test_images, test_labels):
    labels = []
    for model in models:
        # TODO: classify the labels and find unique misclassifications for every model