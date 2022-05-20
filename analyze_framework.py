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


# analyze watermarking framework and verification method
def analyze_framework():
    # Download and prepare the CIFAR10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    print("Test label example: ", test_labels[0])

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    original_seeds = ['037', '096', '117', '197', '216', '243', '328', '349', '398', '477', '530', '544', '622', '710', '718',
             '771', '828', '863', '937', '970']

    extraction_seeds = ['123', '436', '681', '758']

    models = {}
    models_derived_cc = {}
    models_derived_kn = {}

    # load all models
    print("Loading models")
    for seed in original_seeds:
        models[seed] = keras.models.load_model('./new_original_models/model_' + seed + '_new')
        for extraction_seed in extraction_seeds:
            models_derived_cc[seed + '_' + extraction_seed] = keras.models.load_model('./copycatcnn_models/model_' + seed + '_extracted_copycatcnn_' + extraction_seed)
            models_derived_kn[seed + '_' + extraction_seed] = keras.models.load_model('./knockoffnets_models/model_' + seed + '_extracted_knockoffnets_' + extraction_seed)

    print("Finished loading models")

    print("Evaluating performance on seeds:")

    for seed in original_seeds:
        print("=========================== Current seed:", seed, "===================================")

        # get original and derived/unrelated framework/unseen models
        original_model = models[seed]

        derived_framework_seeds = np.random.choice(extraction_seeds, size=1, replace=False)
        unrelated_framework_seeds = np.random.choice([x for x in original_seeds if int(x) != int(seed)], size=3, replace=False)
        print("Derived framework seeds:", derived_framework_seeds)
        print("Unrelated framework seeds:", unrelated_framework_seeds)

        derived_unseen_seeds = [x for x in extraction_seeds if x not in derived_framework_seeds]
        unrelated_unseen_seeds = [x for x in original_seeds if x not in unrelated_framework_seeds]
        print("Derived unseen seeds:", derived_unseen_seeds)
        print("Unrelated unseen seeds:", unrelated_unseen_seeds)

        derived_framework_models = [models_derived_cc[seed + '_' + framework_seed] for framework_seed in derived_framework_seeds]
        unrelated_framework_models = [models[framework_seed] for framework_seed in unrelated_framework_seeds]

        derived_unseen_models = [models_derived_cc[seed + '_' + framework_seed] for framework_seed in derived_unseen_seeds]
        # unrelated_unseen_models = [models[framework_seed] for framework_seed in unrelated_unseen_seeds]

        # generate watermarks with the original and framework models, and CIFAR10 test set
        misclassifications, actual_labels, watermarks, watermark_labels = generate_watermarks(original_model=original_model,
                                                                                              derived_models=derived_framework_models,
                                                                                              unrelated_models=unrelated_framework_models[:-1],
                                                                                              validation_models_unrelated=unrelated_framework_models[-1:],
                                                                                              data=test_images,
                                                                                              true_labels=test_labels)

        # get attacker positive (derived) and negative (unrelated) models (5 of each)
        positive_attacker_models = [models_derived_kn[seed + '_' + attack_seed] for attack_seed in extraction_seeds]
        negative_attacker_seeds = np.random.choice([x for x in unrelated_unseen_seeds], size=5, replace=False)
        negative_attacker_models = [models[attack_seed] for attack_seed in negative_attacker_seeds]

        print("Number of positive attacker models:", len(positive_attacker_models))
        print("Number of negative attacker models:", len(negative_attacker_models))
        print("Negative attacker seeds:", negative_attacker_seeds)

        # get defender unrelated models
        unrelated_unseen_models = [models[defender_seed] for defender_seed in unrelated_unseen_seeds if defender_seed not in negative_attacker_seeds]

        # analyze watermark verification performance
        print("Analyzing verification performance:")
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        print("Positive attacker models:")
        for positive_attacker_model in positive_attacker_models:
            watermarked = is_watermarked(suspect_model=positive_attacker_model,
                                         derived_models=derived_unseen_models,
                                         unrelated_models=unrelated_unseen_models,
                                         original_data=misclassifications,
                                         original_labels=actual_labels,
                                         watermarks=watermarks,
                                         watermark_labels=watermark_labels)

            print("Watermarked:", str(watermarked))

            if watermarked:
                true_positives += 1
            else:
                false_negatives += 1

        print("Negative attacker models:")
        for negative_attacker_model in negative_attacker_models:
            watermarked = is_watermarked(suspect_model=negative_attacker_model,
                                         derived_models=derived_unseen_models,
                                         unrelated_models=unrelated_unseen_models,
                                         original_data=misclassifications,
                                         original_labels=actual_labels,
                                         watermarks=watermarks,
                                         watermark_labels=watermark_labels)

            print("Watermarked:", str(watermarked))

            if watermarked:
                false_positives += 1
            else:
                true_negatives += 1

        # print statistical data
        print("True positives:", true_positives)
        print("True negatives:", true_negatives)
        print("False positives", false_positives)
        print("False negatives", false_negatives)

        print("Original model original accuracy:", original_model.evaluate(misclassifications, watermark_labels)[1])
        print("Original model watermark accuracy:", original_model.evaluate(watermarks, watermark_labels)[1])

        print("Average derived framework model original accuracy:",
              np.mean([derived_framework_model.evaluate(misclassifications, watermark_labels)[1]
                       for derived_framework_model in derived_framework_models]))
        print("Average derived framework model watermark accuracy:",
              np.mean([derived_framework_model.evaluate(watermarks, watermark_labels)[1]
                       for derived_framework_model in derived_framework_models]))

        print("Average unrelated framework model original accuracy:",
              np.mean([unrelated_framework_model.evaluate(misclassifications, watermark_labels)[1]
                       for unrelated_framework_model in unrelated_framework_models]))
        print("Average unrelated framework model watermark accuracy:",
              np.mean([unrelated_framework_model.evaluate(watermarks, watermark_labels)[1]
                       for unrelated_framework_model in unrelated_framework_models]))

        print("Average derived unseen model original accuracy:",
              np.mean([derived_unseen_model.evaluate(misclassifications, watermark_labels)[1]
                       for derived_unseen_model in derived_unseen_models]))
        print("Average derived unseen model watermark accuracy:",
              np.mean([derived_unseen_model.evaluate(watermarks, watermark_labels)[1]
                       for derived_unseen_model in derived_unseen_models]))

        print("Average unrelated unseen model original accuracy:",
              np.mean([unrelated_unseen_model.evaluate(misclassifications, watermark_labels)[1]
                       for unrelated_unseen_model in unrelated_unseen_models]))
        print("Average unrelated unseen model watermark accuracy:",
              np.mean([unrelated_unseen_model.evaluate(watermarks, watermark_labels)[1]
                       for unrelated_unseen_model in unrelated_unseen_models]))

        print("Average positive suspect model original accuracy:",
              np.mean([positive_model.evaluate(misclassifications, watermark_labels)[1]
                       for positive_model in positive_attacker_models]))
        print("Average positive suspect model watermark accuracy:",
              np.mean([positive_model.evaluate(watermarks, watermark_labels)[1]
                       for positive_model in positive_attacker_models]))

        print("Average negative suspect model original accuracy:",
              np.mean([negative_model.evaluate(misclassifications, watermark_labels)[1]
                       for negative_model in negative_attacker_models]))
        print("Average negative suspect model watermark accuracy:",
              np.mean([negative_model.evaluate(watermarks, watermark_labels)[1]
                       for negative_model in negative_attacker_models]))


# run analysis
analyze_framework()
