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
    models[seed] = keras.models.load_model('./model_' + seed)
    models[seed + '_extracted'] = keras.models.load_model('./model_' + seed + '_extracted_half_train')
print('finished loading models')


def find_misclassifications(test_images, test_labels, seeds, models):
    for image, label in zip(test_images, test_labels):
        print(label)
        misclassified_labels = []
        for seed in seeds:
            model = models[seed]
            prediction = model.predict_classes(image.reshape((1, 32, 32, 3)))
            if prediction[0] != label[0]:
                misclassified_labels.append(str((seed, prediction[0])))

            model_extracted = models[seed + '_extracted']
            prediction_extracted = model_extracted.predict_classes(image.reshape((1, 32, 32, 3)))
            if prediction_extracted[0] != label[0]:
                misclassified_labels.append(str((seed + '_extracted', prediction_extracted[0])))

        print('Actual label: ' + str(label[0]))
        print('Misclassified:')
        print(misclassified_labels)
        if len(misclassified_labels) == 2:
            # View image
            plt.imshow(image)
            plt.show()


# find_misclassifications(test_images, test_labels, seeds, models)


def find_average_number_of_classifications(test_images, test_labels, models, seeds):
    num_of_classifications = [0] * len(test_labels)
    predictions = {}

    print('calculating predictions')
    for seed in seeds:
        predictions[seed] = models[seed].predict_classes(test_images)
    print('finished calculating predictions')

    for i in range(len(test_labels)):
        unique_classifications = [0] * 10
        for seed in seeds:
            unique_classifications[predictions[seed][i]] += 1

        num_of_classifications[i] = np.count_nonzero(unique_classifications)

        if num_of_classifications[i] == 8:
            print(unique_classifications)
            for seed2 in seeds:
                print(seed2 + ' : ' + str(predictions[seed2][i]))
            # View image
            plt.imshow(test_images[i])
            plt.show()

    print(num_of_classifications)
    print(max(num_of_classifications))
    return sum(num_of_classifications) / len(num_of_classifications)


# print('Average number of classifications: ' + str(find_average_number_of_classifications(test_images, test_labels, models, seeds)))


def find_boundary_difference(test_images, test_labels, original_model, extracted_model):
    different = 0
    same = 0
    same_correct = 0
    same_incorrect = 0
    original_prediction = original_model.predict_classes(test_images)
    extracted_prediction = extracted_model.predict_classes(test_images)

    for label, original_label, extracted_label in zip(test_labels, original_prediction, extracted_prediction):
        if original_label == extracted_label:
            same += 1
            if original_label == label:
                same_correct += 1
            else:
                same_incorrect += 1
        else:
            different += 1

    return same, different, same_correct, same_incorrect


def find_unique_boundary_size(test_images, test_labels, models, seeds, model_seed):
    print('==============' + model_seed + '================')
    unique_examples = 0
    predictions = {}

    print('calculating predictions')
    for seed in seeds:
        predictions[seed] = models[seed].predict_classes(test_images)
        predictions[seed + '_extracted'] = models[seed + '_extracted'].predict_classes(test_images)
    print('finished calculating predictions')

    original_prediction = predictions[model_seed]
    extracted_prediction = predictions[model_seed + '_extracted']

    for i in range(len(test_labels)):
        unique = True
        if(original_prediction[i] == extracted_prediction[i]) and (original_prediction[i] != test_labels[i]):  # common misclassification
            for other_seed in seeds:
                if other_seed != model_seed:
                    if predictions[other_seed][i] != test_labels[i]:  # not unique
                        unique = False

            if unique:
                unique_examples += 1
                for seed2 in seeds:
                    print(seed2 + ' : ' + str(predictions[seed2][i]))
                # View image
                plt.imshow(test_images[i])
                plt.show()

    return unique_examples


# for seed in seeds:
#     same, different, same_correct, same_incorrect = find_boundary_difference(test_images, test_labels, models[seed], models[seed + '_extracted'])
#     print(seed + ' : ' + str(same) + ', ' + str(different) + ', ' + str(same_correct) + ', ' + str(same_incorrect))

# {'314', '159', '265', '358', '979', '323', '846', '264', '338', '327', '950', '288', '419', '716', '939', '937', '510', '582', '097', '494'}

seed = '950'
u = find_unique_boundary_size(test_images, test_labels, models, seeds, seed)
print(seed + ' : ' + str(u))
