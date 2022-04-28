import art
import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Flatten, InputLayer, Reshape
from keras import datasets, layers, models

# model extraction attacks
from art.attacks import ExtractionAttack
from art.attacks.extraction import CopycatCNN, KnockoffNets, FunctionallyEquivalentExtraction

from art.estimators.classification import KerasClassifier

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# download and prepare the CIFAR10 dataset
print('downloading dataset')
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print('finished downloading dataset')

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# the seeds used for training the models
seeds = {'314', '159', '265', '358', '979', '323', '846', '264', '338', '327', '950', '288', '419', '716', '939', '937', '510', '582', '097', '494'}

# load all models in a dictionary
print('Loading models')
models = {}
for seed in seeds:
    models[seed] = keras.models.load_model('./original_models/model_' + seed)
print('finished loading models')

# split the test data into test and steal datasets
print('Training set size:', str(train_images.shape))
len_steal = 25000
indices = np.random.permutation(len(train_images))
x_steal = train_images[indices[:len_steal]]
y_steal = train_images[indices[:len_steal]]
x_test = test_images  # test_images[indices[len_steal:]]
y_test = test_labels  # test_labels[indices[len_steal:]]

num_epochs = 15


def get_model():
    # Create the convolutional base
    model_base = Sequential()
    model_base.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model_base.add(layers.MaxPooling2D((2, 2)))
    model_base.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_base.add(layers.MaxPooling2D((2, 2)))
    model_base.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add Dense layers on top
    model_base.add(layers.Flatten())
    model_base.add(layers.Dense(64, activation='relu'))
    model_base.add(layers.Dense(10))

    # Compile and train the model
    model_base.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    return model_base


# execute model extraction
print('executing knockoff nets')
for seed in seeds:
    # reset random seeds
    random.seed(314)
    np.random.seed(314)
    tf.random.set_seed(314)

    print('extracting model ' + seed + '(knockoff nets)')
    model = models[seed]
    classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
    attack = KnockoffNets(classifier=classifier_original,
                          batch_size_fit=64,
                          batch_size_query=64,
                          nb_epochs=num_epochs,
                          nb_stolen=len_steal,
                          use_probability=True)
    model_stolen = get_model()
    classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)
    classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
    acc = classifier_stolen._model.evaluate(x_test, y_test)[1]
    print(seed, ":", acc)
    classifier_stolen._model.save('./knockoffnets_models/model_' + seed + '_extracted_knockoffnets')
    print('saved extracted model (knockoff nets)')

print('executing copycatCNN')
for seed in seeds:
    # reset random seeds
    random.seed(314)
    np.random.seed(314)
    tf.random.set_seed(314)

    print('extracting model ' + seed + '(copycatCNN)')
    model = models[seed]
    classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
    attack = CopycatCNN(classifier=classifier_original,
                        batch_size_fit=64,
                        batch_size_query=64,
                        nb_epochs=num_epochs,
                        nb_stolen=len_steal,
                        use_probability=True)
    model_stolen = get_model()
    classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)
    classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
    acc = classifier_stolen._model.evaluate(x_test, y_test)[1]
    print(seed, ":", acc)
    classifier_stolen._model.save('./copycatcnn_models/model_' + seed + '_extracted_copycatcnn')
    print('saved extracted model (copycatCNN)')

