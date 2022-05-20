import random

import numpy as np

import tensorflow as tf

import scipy.stats

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


# generates watermarks for the original model from a given set of data
def generate_watermarks(original_model, derived_models, unrelated_models, validation_models_unrelated, data, true_labels):
    # step 1: get model predictions on data
    preds_original = original_model.predict(data)
    preds_derived = [derived_model.predict(data) for derived_model in derived_models]
    preds_unrelated = [unrelated_model.predict(data) for unrelated_model in unrelated_models]

    # step 2: find data misclassified by original and derived models and correctly classified by unrelated models
    misclassification_indices = [i for i in range(len(data)) if (
            (np.argmax(preds_original[i]) != true_labels[i]) and
            (all(np.argmax(preds_original[i]) == np.argmax(pred_derived[i]) for pred_derived in preds_derived)) and
            (all(true_labels[i] == np.argmax(pred_unrelated[i]) for pred_unrelated in preds_unrelated)))]

    misclassifications = np.array([data[index] for index in misclassification_indices])
    misclassification_labels = np.array([np.argmax(preds_original[index]) for index in misclassification_indices])
    actual_labels = np.array([np.argmax(true_labels[index]) for index in misclassification_indices])

    print('original data length:', len(data))
    print('number of unique misclassifications:', len(misclassifications))
    print('percentage of unique misclassifications:', len(misclassifications) / len(data))

    # step 3: wrap models in classifier wrapping in order to utilize BIM
    classifier = KerasClassifier(original_model, clip_values=(0, 1), use_logits=False)
    classifiers_derived = [KerasClassifier(derived_model, clip_values=(0, 1), use_logits=False) for
                           derived_model in derived_models]
    classifiers_unrelated = [KerasClassifier(unrelated_model, clip_values=(0, 1), use_logits=False) for
                             unrelated_model in unrelated_models]

    print("Generating watermarks:")

    # step 4: generate watermark permutations with BIM

    # step 4.1: direct toward misclassification on original model
    watermarks_original = BasicIterativeMethod(estimator=classifier,
                                               eps=0.05,
                                               eps_step=0.01,
                                               max_iter=100,
                                               targeted=True,
                                               batch_size=32,
                                               verbose=False).generate(x=misclassifications, y=misclassification_labels)

    # step 4.2: direct toward misclassification on derived models
    watermarks_derived = [BasicIterativeMethod(estimator=classifier_derived,
                                               eps=0.05,
                                               eps_step=0.01,
                                               max_iter=100,
                                               targeted=True,
                                               batch_size=32,
                                               verbose=False).generate(x=misclassifications, y=misclassification_labels)
                          for classifier_derived in classifiers_derived]

    # step 4.3: direct towards correct classification on unrelated models
    watermarks_unrelated = [BasicIterativeMethod(estimator=classifier_unrelated,
                                                 eps=0.05,
                                                 eps_step=0.01,
                                                 max_iter=100,
                                                 targeted=True,
                                                 batch_size=32,
                                                 verbose=False).generate(x=misclassifications, y=actual_labels)
                            for classifier_unrelated in classifiers_unrelated]

    # step 5: calculate mean of generated permutations

    # step 5.1: calculate mean of misclassified (original + derived) model watermarks
    watermarks_misclassified = np.mean(np.array([watermarks_original, watermarks_derived]), axis=0)

    # step 5.2: calculate mean of correct (unrelated) model watermarks
    watermarks_correct = np.mean(np.array(watermarks_unrelated), axis=0)

    # step 5.3: take the mean of misclassified and correct directed watermarks as the watermarks
    watermarks = np.average(np.array([watermarks_misclassified, watermarks_correct]), axis=0, weights=[0.5, 0.5])

    # step 6: validation testing - remove watermarks that aren't correctly classified by holdout unrelated models
    preds_validation_unrelated = [validation_model_unrelated.predict(watermarks) for validation_model_unrelated in validation_models_unrelated]

    watermark_indices = [i for i in range(len(watermarks)) if (
        all(np.argmax(pred_validation_unrelated[i]) == actual_labels[i]
            for pred_validation_unrelated in preds_validation_unrelated)
    )]

    misclassifications = np.array([misclassifications[index] for index in watermark_indices])
    actual_labels = np.array([actual_labels[index] for index in watermark_indices])
    watermarks = np.array([watermarks[index] for index in watermark_indices])
    watermark_labels = np.array([misclassification_labels[index] for index in watermark_indices])

    print("Number of watermarks:", len(watermarks))

    return misclassifications, actual_labels, watermarks, watermark_labels


def is_watermarked(suspect_model, derived_models, unrelated_models, original_data, original_labels, watermarks, watermark_labels):

    # step 1: calculate original and watermark accuracy for suspect, derived, and unrelated models
    original_acc_suspect = suspect_model.evaluate(original_data, watermark_labels)[1]
    watermark_acc_suspect = suspect_model.evaluate(watermarks, watermark_labels)[1]

    original_accs_derived = [derived_model.evaluate(original_data, watermark_labels)[1]
                            for derived_model in derived_models]
    watermark_accs_derived = [derived_model.evaluate(watermarks, watermark_labels)[1]
                             for derived_model in derived_models]

    original_accs_unrelated = [unrelated_model.evaluate(original_data, watermark_labels)[1]
                               for unrelated_model in unrelated_models]
    watermark_accs_unrelated = [unrelated_model.evaluate(watermarks, watermark_labels)[1]
                               for unrelated_model in unrelated_models]

    # step 2: generate normal distributions for derived and unrelated accuracies
    original_distribution_derived = scipy.stats.norm(np.mean(original_accs_derived), np.std(original_accs_derived))
    watermarked_distribution_derived = scipy.stats.norm(np.mean(watermark_accs_derived), np.std(watermark_accs_derived))

    original_distribution_unrelated = scipy.stats.norm(np.mean(original_accs_unrelated), np.std(original_accs_unrelated))
    watermarked_distribution_unrelated = scipy.stats.norm(np.mean(watermark_accs_unrelated), np.std(watermark_accs_unrelated))

    # step 3: calculate pdf for suspect accuracies in each distribution
    pdf_original_derived = original_distribution_derived.pdf(original_acc_suspect)
    pdf_watermarked_derived = watermarked_distribution_derived.pdf(watermark_acc_suspect)

    pdf_original_unrelated = original_distribution_unrelated.pdf(original_acc_suspect)
    pdf_watermarked_unrelated = watermarked_distribution_unrelated.pdf(watermark_acc_suspect)

    # step 4: suspect is considered watermarked if pdf for both accuracies is higher in derived distributions
    return (pdf_original_derived > pdf_original_unrelated) and (pdf_watermarked_derived > pdf_watermarked_unrelated)

