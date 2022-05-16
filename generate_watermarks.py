import random

import numpy as np

import tensorflow as tf

import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Flatten, InputLayer, Reshape
from keras import datasets, layers, models
import matplotlib.pyplot as plt

from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import KerasClassifier

if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# Download and prepare the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Test label example: ", test_labels[0])

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

seeds = ['037', '096', '117', '197', '216', '243', '328', '349', '398', '477', '530', '544', '622', '710', '718', '771', '828', '863', '937', '970']

models = {}
models_cc = {}
models_kn = {}

# load all models
print("Loading models")
for seed in seeds:
    models[seed] = keras.models.load_model('./new_original_models/model_' + seed + '_new')
    models_cc[seed] = keras.models.load_model('./copycatcnn_models/model_' + seed + '_extracted_copycatcnn')
    models_kn[seed] = keras.models.load_model('./knockoffnets_models/model_' + seed + '_extracted_knockoffnets')

print("Finished loading models")

print("Evaluating performance on seeds:")

for seed in seeds:
    print("=========================== Current seed:", seed, "===================================")

    model = models[seed]
    model_cc = models_cc[seed]
    model_kn = models_kn[seed]

    unrelated_seeds = np.random.choice([x for x in seeds if int(x) != int(seed)], size=2, replace=False)
    print("Unrelated seeds:", unrelated_seeds)

    unrelated_models = [models[unrelated_seed] for unrelated_seed in unrelated_seeds]

    # currently, the copycatcnn model is used as the derived model for watermark generation
    y_pred_original = model.predict(test_images)
    y_pred_derived = model_cc.predict(test_images)

    y_pred_unrelated = [unrelated_model.predict(test_images) for unrelated_model in unrelated_models]

    misclassifications = np.array([test_images[i] for i in range(len(test_images)) if ((np.argmax(y_pred_original[i]) != test_labels[i]) and (np.argmax(y_pred_original[i]) == np.argmax(y_pred_derived[i])) and (test_labels[i] == np.argmax(y_pred_unrelated[0][i])) and (test_labels[i] == np.argmax(y_pred_unrelated[1][i])))])
    misclassification_labels = np.array([np.argmax(y_pred_original[i]) for i in range(len(test_images)) if ((np.argmax(y_pred_original[i]) != test_labels[i]) and (np.argmax(y_pred_original[i]) == np.argmax(y_pred_derived[i])) and (test_labels[i] == np.argmax(y_pred_unrelated[0][i])) and (test_labels[i] == np.argmax(y_pred_unrelated[1][i])))])
    actual_labels = np.array([test_labels[i] for i in range(len(test_images)) if ((np.argmax(y_pred_original[i]) != test_labels[i]) and (np.argmax(y_pred_original[i]) == np.argmax(y_pred_derived[i])) and (test_labels[i] == np.argmax(y_pred_unrelated[0][i])) and (test_labels[i] == np.argmax(y_pred_unrelated[1][i])))])

    print('number of unique misclassifications: ', len(misclassifications))

    # image = test_images[0]
    # print(image.shape)
    # label = test_labels[0]
    #
    # plt.imshow(image)
    # plt.show()
    #
    # print('original label: ', test_labels[0])
    # print('watermarked label: ', test_labels[1])

    classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
    classifier_derived = KerasClassifier(model_cc, clip_values=(0, 1), use_logits=False)
    classifiers_unrelated = [KerasClassifier(models[unrelated_seed], clip_values=(0, 1), use_logits=False) for unrelated_seed in unrelated_seeds]

    # watermark = BasicIterativeMethod(   estimator=classifier,
    #                                     eps=0.05,
    #                                     eps_step=0.01,
    #                                     max_iter=100,
    #                                     targeted=True,
    #                                     batch_size=32,
    #                                     verbose=True).generate(x=test_images[:1], y=test_labels[1])

    print("Generating watermarks:")

    # direct toward misclassification on original model
    watermarks_original = BasicIterativeMethod(   estimator=classifier,
                                        eps=0.05,
                                        eps_step=0.01,
                                        max_iter=100,
                                        targeted=True,
                                        batch_size=32,
                                        verbose=False).generate(x=misclassifications, y=misclassification_labels)

    # direct toward misclassification on derived model
    watermarks_derived = BasicIterativeMethod(estimator=classifier_derived,
                                      eps=0.05,
                                      eps_step=0.01,
                                      max_iter=100,
                                      targeted=True,
                                      batch_size=32,
                                      verbose=False).generate(x=misclassifications, y=misclassification_labels)

    # direct towards correct classification on unrelated models
    watermarks_unrelated = [BasicIterativeMethod(estimator=classifier_unrelated,
                                      eps=0.05,
                                      eps_step=0.01,
                                      max_iter=100,
                                      targeted=True,
                                      batch_size=32,
                                      verbose=False).generate(x=misclassifications, y=actual_labels) for classifier_unrelated in classifiers_unrelated]

    # calculate mean of misclassified (original + derived) model watermarks
    watermarks_misclassified = np.mean(np.array([watermarks_original, watermarks_derived]), axis=0)

    # calculate mean of correct (unrelated) model watermarks
    watermarks_correct = np.mean(np.array(watermarks_unrelated), axis=0)

    # take the mean of misclassified and correct directed watermarks as the watermarks
    watermarks = np.mean(np.array([watermarks_misclassified, watermarks_correct]), axis=0)


    # validation testing - remove watermarks that aren't correctly classified by a holdout derived and unrelated model
    validation_seed = np.random.choice([x for x in seeds if (int(x) != int(seed) and x not in unrelated_seeds)], size=1, replace=False)[0]
    print("Validation model:", validation_seed)

    validation_model = models[validation_seed]

    validation_pred_unrelated = validation_model.predict(watermarks)
    validation_pred_derived = model_kn.predict(watermarks)

    misclassifications = np.array([misclassifications[i] for i in range(len(misclassifications)) if (
                np.argmax(validation_pred_unrelated[i]) == actual_labels[i])])
    watermarks = np.array([watermarks[i] for i in range(len(watermarks)) if (
                np.argmax(validation_pred_unrelated[i]) == actual_labels[i])])
    misclassification_labels = np.array([misclassification_labels[i] for i in range(len(misclassification_labels)) if (
                np.argmax(validation_pred_unrelated[i]) == actual_labels[i])])

    print("Number of watermarks:", len(watermarks))

    print("Finished generating watermarks:")

    print("Evaluating on derived models:")

    original_acc = model.evaluate(misclassifications, misclassification_labels)[1]
    watermark_acc = model.evaluate(watermarks, misclassification_labels)[1]
    print("Original model - original misclassification correspondence:", str(original_acc),
          ", watermark correspondence:", str(watermark_acc))

    original_acc_cc = model_cc.evaluate(misclassifications, misclassification_labels)[1]
    watermark_acc_cc = model_cc.evaluate(watermarks, misclassification_labels)[1]
    print("CopycatCNN - original misclassification correspondence:", str(original_acc_cc), ", watermark correspondence:", str(watermark_acc_cc))

    original_acc_kn = model_kn.evaluate(misclassifications, misclassification_labels)[1]
    watermark_acc_kn = model_kn.evaluate(watermarks, misclassification_labels)[1]
    print("Knockoff Nets - original misclassification correspondence:", str(original_acc_kn),
          ", watermark correspondence:", str(watermark_acc_kn))

    print("Finished evaluating on derived models")

    print("Evaluating on unrelated models:")

    for unrelated_seed in seeds:
        if int(unrelated_seed) != int(seed):
            unrelated_model = models[unrelated_seed]
            original_acc_unrelated = unrelated_model.evaluate(misclassifications, misclassification_labels)[1]
            watermark_acc_unrelated = unrelated_model.evaluate(watermarks, misclassification_labels)[1]
            print("Seed:", unrelated_seed, ", original misclassification correspondence:", str(original_acc_unrelated),
                  ", watermark correspondence:", str(watermark_acc_unrelated))

    print("Finished evaluating on unrelated models")

    # if int(seed) == 117:
    #     plt.imshow(misclassifications[0])
    #     plt.show()
    #
    #     plt.imshow(watermarks[0])
    #     plt.show()

    # print(watermark.shape)
    # plt.imshow(watermark[0])
    # plt.show()
    # prediction = model.predict(watermark)
    # print(np.argmax(prediction[0]))

print("Finished evaluating performance on seeds")
