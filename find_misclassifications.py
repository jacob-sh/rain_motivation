import random
random.seed(20)

import numpy as np
np.random.seed(20)

import tensorflow as tf
tf.random.set_seed(20)

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from scipy.special import softmax

from art.attacks.evasion import BasicIterativeMethod
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
# seeds = {'314', '159', '265', '358', '979', '323', '846', '264', '338', '327', '950', '288', '419', '716', '939', '937', '510', '582', '097', '494'}

# load all models in a dictionary
print('Loading models')
original_seeds = ['037', '096', '117', '197', '216', '243', '328', '349', '398', '477', '530', '544', '622', '710', '718',
             '771', '828', '863', '937', '970']

extraction_seeds = ['123', '295', '436', '681', '758']

models = {}
# derived_models = {}

# load all original models
print("Loading models")
for seed in original_seeds:
    models[seed] = keras.models.load_model('./new_original_models/model_' + seed + '_new')
    # derived_models[seed + '_extracted_1'] = keras.models.load_model('./copycatcnn_models/model_' + seed + '_extracted_copycatcnn_' + '123')
    # derived_models[seed + '_extracted_2'] = keras.models.load_model('./copycatcnn_models/model_' + seed + '_extracted_copycatcnn_' + '681')
print('finished loading models')


def database_analysis(models, derived_models, original_seeds, data, true_labels):
    dbsize = []
    disagreements = []
    total_unique = []
    avg_unique = []
    total_transfer = []
    avg_transfer = []

    # total number of transferable instances and confidence levels for the applied model (whose seed is the variable seed)
    total_applied_model_transferables = []
    total_applied_model_trans_confidences = []

    for seed in original_seeds:
        print('============', seed, 'iteration =============================')
        # print data length
        print('Data length:', len(data))

        print("Performing BIM")
        classifier = KerasClassifier(models[seed], clip_values=(0, 1), use_logits=False)
        data = BasicIterativeMethod(estimator=classifier,
                                           eps=0.01,
                                           eps_step=0.001,
                                           max_iter=20,
                                           targeted=True,
                                           batch_size=32,
                                           verbose=False).generate(x=data,
                                                                   y=models[seed].predict(data))
        print("Finished Performing BIM")

        preds_per_seed = {}
        for original_seed in original_seeds:
            preds_per_seed[original_seed] = models[original_seed].predict(data)
            preds_per_seed[original_seed + '_extracted'] = derived_models[original_seed + '_extracted'].predict(data)

        preds_original = [preds_per_seed[seed] for seed in original_seeds]
        # preds_derived = [derived_model.predict(data) for derived_model in derived_models]

        # find disagreements (data misclassified by at least one original model)
        disagreement_indices = [i for i in range(len(data)) if (
                (any([np.argmax(pred_original[i]) != true_labels[i] for pred_original in preds_original])))]
        disagreement_data = np.array([np.array(data[i]).reshape((32, 32, 3)) for i in disagreement_indices])
        disagreement_labels = np.array([true_labels[i] for i in disagreement_indices])

        print(data.shape)
        print(disagreement_data.shape)

        # print("Performing BIM")
        # classifier = KerasClassifier(models[seed], clip_values=(0, 1), use_logits=False)
        # disagreement_data = BasicIterativeMethod(estimator=classifier,
        #                                    eps=0.01,
        #                                    eps_step=0.001,
        #                                    max_iter=20,
        #                                    targeted=True,
        #                                    batch_size=32,
        #                                    verbose=False).generate(x=disagreement_data,
        #                                                            y=models[seed].predict(disagreement_data))
        # print("Finished Performing BIM")

        # update predictions
        preds_per_seed = {}
        for original_seed in original_seeds:
            preds_per_seed[original_seed] = models[original_seed].predict(disagreement_data)
            preds_per_seed[original_seed + '_extracted'] = derived_models[original_seed + '_extracted'].predict(disagreement_data)

        print('Disagreements length:', len(disagreement_indices))

        unique_misclassifications = []
        unique_data = []
        unique_labels = []
        # find unique classifications (correctly classified by one model)
        for original_seed in original_seeds:
            print('============', original_seed, 'unique misclassifications =============================')
            preds_current_seed = preds_per_seed[original_seed]
            preds_unrelated = [preds_per_seed[unrelated_seed] for unrelated_seed in original_seeds if int(unrelated_seed) != int(original_seed)]
            unique_indices = [i for i in range(len(disagreement_data)) if (
                (np.argmax(preds_current_seed[i]) != disagreement_labels[i]) and
                all([np.argmax(pred_unrelated[i]) == disagreement_labels[i] for pred_unrelated in preds_unrelated]))]
            print("Unique misclassifications:", len(unique_indices))
            unique_misclassifications.append(len(unique_indices))
            unique_data.extend([np.array(disagreement_data[i]).reshape((32, 32, 3)) for i in unique_indices])
            unique_labels.extend([disagreement_labels[i] for i in unique_indices])

        unique_data = np.array(unique_data)
        unique_labels = np.array(unique_labels)

        print("Total unique misclassifications:", sum(unique_misclassifications))
        print("Average unique misclassifications per model:", sum(unique_misclassifications) / len(unique_misclassifications))

        # print("Performing BIM")
        # classifier = KerasClassifier(models[seed], clip_values=(0, 1), use_logits=False)
        # unique_data = BasicIterativeMethod(estimator=classifier,
        #                                    eps=0.01,
        #                                    eps_step=0.001,
        #                                    max_iter=20,
        #                                    targeted=True,
        #                                    batch_size=32,
        #                                    verbose=False).generate(x=unique_data,
        #                                                            y=models[seed].predict(unique_data))
        # print("Finished Performing BIM")

        # update predictions
        preds_per_seed = {}
        for original_seed in original_seeds:
            preds_per_seed[original_seed] = models[original_seed].predict(unique_data)
            preds_per_seed[original_seed + '_extracted'] = derived_models[original_seed + '_extracted'].predict(
                unique_data)

        transfer_misclassifications = []
        # find transferable classifications (correctly classified by one model and its derived model)
        for original_seed in original_seeds:
            print('============', original_seed, 'transferrable misclassifications =============================')
            preds_current_seed = preds_per_seed[original_seed]
            preds_derived = preds_per_seed[original_seed + '_extracted']
            preds_unrelated = [preds_per_seed[unrelated_seed] for unrelated_seed in original_seeds if
                               int(unrelated_seed) != int(original_seed)]
            transfer_indices = [i for i in range(len(unique_data)) if (
                    (np.argmax(preds_current_seed[i]) != unique_labels[i]) and
                    (np.argmax(preds_current_seed[i]) == np.argmax(preds_derived[i])) and
                    all([np.argmax(pred_unrelated[i]) == unique_labels[i] for pred_unrelated in preds_unrelated]))]
            print("Transferred misclassifications:", len(transfer_indices))
            transfer_misclassifications.append(len(transfer_indices))

            if int(original_seed) == int(seed):  # applied model
                print("Number of applied model transferable instances:", len(transfer_indices))
                total_applied_model_transferables.append(len(transfer_indices))
                print("Example confidence array:", str(softmax(preds_current_seed[0])))
                confidences = np.array([np.max(softmax(preds_current_seed[i])) for i in range(len(unique_data)) if i in transfer_indices])
                print("Number of confidences:", len(confidences))
                print("Average applied transferable confidence", np.mean(confidences))
                total_applied_model_trans_confidences.append(np.mean(confidences))

        print("Total transfer misclassifications:", sum(transfer_misclassifications))
        print("Average transfer misclassifications per model:",
              sum(transfer_misclassifications) / len(transfer_misclassifications))

        # calculate transferable instances confidence level

        dbsize.append(len(data))
        disagreements.append(len(disagreement_indices))
        total_unique.append(sum(unique_misclassifications))
        avg_unique.append(sum(unique_misclassifications) / len(unique_misclassifications))
        total_transfer.append(sum(transfer_misclassifications))
        avg_transfer.append(sum(transfer_misclassifications) / len(transfer_misclassifications))

    # return averages from 20 trials
    return sum(dbsize) / len(dbsize), sum(disagreements) / len(disagreements), sum(total_unique) / len(total_unique), sum(avg_unique) / len(avg_unique), sum(total_transfer) / len(total_transfer), sum(avg_transfer) / len(avg_transfer), sum(total_applied_model_transferables) / len(total_applied_model_transferables), sum(total_applied_model_trans_confidences) / len(total_applied_model_trans_confidences)


# dbsize, disagreements, total_unique, avg_unique, total_transfer, avg_transfer, total_applied_trans, avg_confidences = database_analysis(models, derived_models, original_seeds, test_images, test_labels)
# print("======================================================")
# print("Database size:", dbsize)
# print("Disagreements size:", disagreements)
# print("Total unique misclassifications:", total_unique)
# print("Average unique misclassifications:", avg_unique)
# print("Total transfer misclassifications:", total_transfer)
# print("Average transfer misclassifications:", avg_transfer)
# print("Average applied transferable instances:", total_applied_trans)
# print("Average applied transferable instance confidence:", avg_confidences)
# print("======================================================")


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

# seed = '950'
# u = find_unique_boundary_size(test_images, test_labels, models, seeds, seed)
# print(seed + ' : ' + str(u))


# =============================================== RAIN Framework Analysis ==============================================


def rain_framework(model, derived_models, unrelated_models, data, true_labels):

    # print data length
    print('Data length:', len(data))

    # Step 1: get data predictions for every model
    original_predictions_no_softmax = model.predict(data)
    derived_predictions_no_softmax = np.array([derived_model.predict(data) for derived_model in derived_models])
    unrelated_predictions_no_softmax = np.array([unrelated_model.predict(data) for unrelated_model in unrelated_models])

    # apply softmax to predictions
    original_predictions = np.array([softmax(original_prediction) for original_prediction in original_predictions_no_softmax])
    derived_predictions = np.array([np.array([softmax(derived_prediction) for derived_prediction in model_prediction]) for model_prediction in derived_predictions_no_softmax])
    unrelated_predictions = np.array([np.array([softmax(unrelated_prediction) for unrelated_prediction in model_prediction]) for model_prediction in unrelated_predictions_no_softmax])

    # Step 2: find disagreements (data misclassified by at least one original model)
    disagreement_indices = [i for i in range(len(data)) if (
            (any([np.argmax(unrelated_prediction[i]) != true_labels[i] for unrelated_prediction in unrelated_predictions])) or
            (np.argmax(original_predictions[i]) != true_labels[i]))]
    disagreements = np.array([np.array(data[i]).reshape(data[i].shape) for i in disagreement_indices])
    disagreement_true_labels = np.array([true_labels[i] for i in disagreement_indices])

    print('Data shape:', data.shape)
    print('Disagreements shape:', disagreements.shape)

    # Step 3: perform BIM on disagreements
    print("Performing BIM on disagreements")
    classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
    disagreements = BasicIterativeMethod(estimator=classifier,
                                       eps=0.01,
                                       eps_step=0.001,
                                       max_iter=20,
                                       targeted=True,
                                       batch_size=32,
                                       verbose=False).generate(x=disagreements,
                                                               y=model.predict(disagreements))
    print("Finished Performing BIM on disagreements")

    # Step 4: update predictions
    original_predictions_no_softmax = model.predict(disagreements)
    derived_predictions_no_softmax = np.array([derived_model.predict(disagreements) for derived_model in derived_models])
    unrelated_predictions_no_softmax = np.array([unrelated_model.predict(disagreements) for unrelated_model in unrelated_models])

    # apply softmax to predictions
    original_predictions = np.array(
        [softmax(original_prediction) for original_prediction in original_predictions_no_softmax])
    derived_predictions = np.array(
        [np.array([softmax(derived_prediction) for derived_prediction in model_prediction]) for model_prediction in
         derived_predictions_no_softmax])
    unrelated_predictions = np.array(
        [np.array([softmax(unrelated_prediction) for unrelated_prediction in model_prediction]) for model_prediction
         in unrelated_predictions_no_softmax])

    print('Total disagreements:', len(disagreement_indices))

    # Step 5: find unique disagreements (correctly misclassified by original model only)
    unique_indices = [i for i in range(len(disagreements)) if (
            (all([np.argmax(unrelated_prediction[i]) == disagreement_true_labels[i] for unrelated_prediction in
                  unrelated_predictions])) and
            (np.argmax(original_predictions[i]) != disagreement_true_labels[i]))]
    unique_disagreements = np.array([np.array(disagreements[i]).reshape(disagreements[i].shape) for i in unique_indices])
    unique_disagreement_true_labels = np.array([disagreement_true_labels[i] for i in unique_indices])

    print("Total unique disagreements:", len(unique_disagreements))

    # Step 6: update predictions
    original_predictions_no_softmax = model.predict(unique_disagreements)
    derived_predictions_no_softmax = np.array(
        [derived_model.predict(unique_disagreements) for derived_model in derived_models])
    unrelated_predictions_no_softmax = np.array(
        [unrelated_model.predict(unique_disagreements) for unrelated_model in unrelated_models])

    # apply softmax to predictions
    original_predictions = np.array(
        [softmax(original_prediction) for original_prediction in original_predictions_no_softmax])
    derived_predictions = np.array(
        [np.array([softmax(derived_prediction) for derived_prediction in model_prediction]) for model_prediction in
         derived_predictions_no_softmax])
    unrelated_predictions = np.array(
        [np.array([softmax(unrelated_prediction) for unrelated_prediction in model_prediction]) for model_prediction
         in unrelated_predictions_no_softmax])

    # Step 7: find transferable disagreements (identically misclassified by original model and its derived models)
    transferable_indices = [i for i in range(len(unique_disagreements)) if (
            (all([np.argmax(derived_prediction[i]) == np.argmax(original_predictions[i]) for derived_prediction in
                  derived_predictions])) and
            (np.argmax(original_predictions[i]) != unique_disagreement_true_labels[i]))]
    transferable_disagreements = np.array(
        [np.array(unique_disagreements[i]).reshape(unique_disagreements[i].shape) for i in transferable_indices])
    transferable_disagreement_true_labels = np.array([unique_disagreement_true_labels[i] for i in transferable_indices])

    print("Total transferable disagreements:", len(transferable_indices))

    # Step 8: update predictions
    original_predictions_no_softmax = model.predict(transferable_disagreements)
    derived_predictions_no_softmax = np.array(
        [derived_model.predict(transferable_disagreements) for derived_model in derived_models])
    unrelated_predictions_no_softmax = np.array(
        [unrelated_model.predict(transferable_disagreements) for unrelated_model in unrelated_models])

    # apply softmax to predictions
    original_predictions = np.array(
        [softmax(original_prediction) for original_prediction in original_predictions_no_softmax])
    derived_predictions = np.array(
        [np.array([softmax(derived_prediction) for derived_prediction in model_prediction]) for model_prediction in
         derived_predictions_no_softmax])
    unrelated_predictions = np.array(
        [np.array([softmax(unrelated_prediction) for unrelated_prediction in model_prediction]) for model_prediction
         in unrelated_predictions_no_softmax])



    # =============================================== Statistics ===================================================

    watermark_labels = np.argmax(original_predictions, axis=1)

    # calculate transferable disagreements accuracy
    original_true_acc = model.evaluate(transferable_disagreements, transferable_disagreement_true_labels)[1]
    original_watermark_acc = model.evaluate(transferable_disagreements, watermark_labels)[1]

    derived_true_acc = np.array([derived_model.evaluate(transferable_disagreements, transferable_disagreement_true_labels)[1] for derived_model in derived_models])
    derived_watermark_acc = np.array([derived_model.evaluate(transferable_disagreements, watermark_labels)[1] for derived_model in derived_models])

    unrelated_true_acc = np.array(
        [unrelated_model.evaluate(transferable_disagreements, transferable_disagreement_true_labels)[1] for
         unrelated_model in unrelated_models])
    unrelated_watermark_acc = np.array(
        [unrelated_model.evaluate(transferable_disagreements, watermark_labels)[1] for unrelated_model in
         unrelated_models])

    print('Seen model true and watermark accuracies:')
    print('Original model', original_true_acc, original_watermark_acc)
    print('Derived models', derived_true_acc, derived_watermark_acc)
    print('Unrelated models', unrelated_true_acc, unrelated_watermark_acc)

    # calculate transferable disagreements confidence
    original_confidence = np.mean([np.max(prediction) for prediction in original_predictions])
    derived_confidence = np.mean([[np.max(prediction) for prediction in derived_prediction] for derived_prediction in derived_predictions])
    unrelated_confidence = np.mean(
        [[np.max(prediction) for prediction in unrelated_prediction] for unrelated_prediction in unrelated_predictions])

    print('Seen model confidences:')
    print('Original model', original_confidence)
    print('Derived models', derived_confidence)
    print('Unrelated models', unrelated_confidence)

    # return watermarks
    return transferable_disagreements, watermark_labels, transferable_disagreement_true_labels, original_confidence, derived_confidence, unrelated_confidence


def measure_performance(watermarks, watermark_labels, true_labels, derived_models, unrelated_models):
    # make predictions
    derived_predictions_no_softmax = np.array(
        [derived_model.predict(watermarks) for derived_model in derived_models])
    unrelated_predictions_no_softmax = np.array(
        [unrelated_model.predict(watermarks) for unrelated_model in unrelated_models])

    # apply softmax to predictions
    derived_predictions = np.array(
        [np.array([softmax(derived_prediction) for derived_prediction in model_prediction]) for model_prediction in
         derived_predictions_no_softmax])
    unrelated_predictions = np.array(
        [np.array([softmax(unrelated_prediction) for unrelated_prediction in model_prediction]) for model_prediction
         in unrelated_predictions_no_softmax])

    # measure accuracy
    derived_watermark_acc = np.array(
        [derived_model.evaluate(watermarks, watermark_labels)[1] for derived_model in derived_models])

    unrelated_watermark_acc = np.array(
        [unrelated_model.evaluate(watermarks, watermark_labels)[1] for unrelated_model in unrelated_models])

    avg_derived_acc = np.mean(derived_watermark_acc)
    avg_unrelated_acc = np.mean(unrelated_watermark_acc)

    # measure watermark and true label confidence
    derived_watermark_confidence = np.mean([[np.max(derived_prediction[i]) for i in range(len(watermark_labels))
                                             if np.argmax(derived_prediction[i]) == watermark_labels[i]
                                             ] for derived_prediction in derived_predictions])

    avg_derived_watermark_confidence = np.mean(derived_watermark_confidence)

    derived_true_confidence = np.mean([[np.max(derived_prediction[i]) for i in range(len(watermark_labels))
                                             if np.argmax(derived_prediction[i]) == true_labels[i]
                                             ] for derived_prediction in derived_predictions])

    avg_derived_true_confidence = np.mean(derived_true_confidence)

    unrelated_watermark_confidence = np.mean([[np.max(unrelated_prediction[i]) for i in range(len(watermark_labels))
                                             if np.argmax(unrelated_prediction[i]) == watermark_labels[i]
                                             ] for unrelated_prediction in unrelated_predictions])

    avg_unrelated_watermark_confidence = np.mean(unrelated_watermark_confidence)

    unrelated_true_confidence = np.mean([[np.max(unrelated_prediction[i]) for i in range(len(watermark_labels))
                                        if np.argmax(unrelated_prediction[i]) == true_labels[i]
                                        ] for unrelated_prediction in unrelated_predictions])

    avg_unrelated_true_confidence = np.mean(unrelated_true_confidence)

    return avg_derived_acc, avg_unrelated_acc, avg_derived_watermark_confidence, avg_derived_true_confidence, avg_unrelated_watermark_confidence, avg_unrelated_true_confidence


original_seen_watermark_confidences = []
derived_seen_watermark_confidences = []
unrelated_seen_true_confidences = []

derived_unseen_watermark_confidences = []
unrelated_unseen_watermark_confidences = []

derived_unseen_true_confidences = []
unrelated_unseen_true_confidences = []

derived_watermark_accs = []
unrelated_watermark_accs = []

watermark_sizes = []

for seed in ['828', '863', '937', '970']:  # original_seeds:
    print("=============================", "Evaluating on seed:", seed, "=============================")

    # load model
    model = models[seed]

    # load derived models
    derived_models = {}
    for extraction_seed in extraction_seeds:
        derived_models[extraction_seed + '_cnn'] = keras.models.load_model('./copycatcnn_models/model_' + seed + '_extracted_copycatcnn_' + extraction_seed)
        derived_models[extraction_seed + '_kon'] = keras.models.load_model('./knockoffnets_models/model_' + seed + '_extracted_knockoffnets_' + extraction_seed)

    # divide derived and unrelated models into seen and unseen
    derived_seen_seeds = np.random.choice(extraction_seeds, size=3, replace=False)
    unrelated_seen_seeds = np.random.choice([x for x in original_seeds if int(x) != int(seed)], size=3,
                                                 replace=False)
    print("Derived seen seeds:", derived_seen_seeds)
    print("Unrelated seen seeds:", unrelated_seen_seeds)

    derived_unseen_seeds = [x for x in extraction_seeds if x not in derived_seen_seeds]
    unrelated_unseen_seeds = [x for x in original_seeds if
                              ((x not in unrelated_seen_seeds) and (int(x) != int(seed)))]
    print("Derived unseen seeds:", derived_unseen_seeds)
    print("Unrelated unseen seeds:", unrelated_unseen_seeds)

    # generate watermarks with seen models
    derived_seen_models = [derived_models[seen_seed + '_cnn'] for seen_seed in derived_seen_seeds]
    unrelated_seen_models = [models[seen_seed] for seen_seed in unrelated_seen_seeds]

    watermarks, watermark_labels, true_labels, original_confidence, derived_confidence, unrelated_confidence = rain_framework(model=model,
                                        derived_models=derived_seen_models,
                                        unrelated_models=unrelated_seen_models,
                                        data=test_images,
                                        true_labels=test_labels)

    # determine transferability to unseen derived/unrelated models
    derived_unseen_models = np.append([derived_models[seen_seed + '_cnn'] for seen_seed in derived_unseen_seeds],
                                      [[derived_models[extraction_seed + '_kon'] for extraction_seed in extraction_seeds]])

    unrelated_unseen_models = [models[unseen_seed] for unseen_seed in unrelated_unseen_seeds]

    avg_derived_acc, avg_unrelated_acc, avg_derived_watermark_confidence, avg_derived_true_confidence, \
    avg_unrelated_watermark_confidence, avg_unrelated_true_confidence = measure_performance(watermarks=watermarks,
                                                                                           watermark_labels=watermark_labels,
                                                                                           true_labels=true_labels,
                                                                                           derived_models=derived_unseen_models,
                                                                                           unrelated_models=unrelated_unseen_models)

    # append new values
    original_seen_watermark_confidences.append(original_confidence)
    derived_seen_watermark_confidences.append(derived_confidence)
    unrelated_seen_true_confidences.append(unrelated_confidence)

    derived_unseen_watermark_confidences.append(avg_derived_watermark_confidence)
    unrelated_unseen_watermark_confidences.append(avg_unrelated_watermark_confidence)

    derived_unseen_true_confidences.append(avg_derived_true_confidence)
    unrelated_unseen_true_confidences.append(avg_unrelated_true_confidence)

    derived_watermark_accs.append(avg_derived_acc)
    unrelated_watermark_accs.append(avg_unrelated_acc)

    watermark_sizes.append(watermarks.shape[0])

    print("Seen confidence values:", original_confidence, derived_confidence, unrelated_confidence)
    print("Watermark size:", watermarks.shape[0])
    print("Other statistics:", avg_derived_acc, avg_unrelated_acc, avg_derived_watermark_confidence,
          avg_derived_true_confidence, avg_unrelated_watermark_confidence, avg_unrelated_true_confidence)

# print average values
print('Original watermark confidences', np.mean(original_seen_watermark_confidences))
print('Derived seen watermark confidences', np.mean(derived_seen_watermark_confidences))
print('Unrelated seen true confidences', np.mean(unrelated_seen_true_confidences))

print('Derived unseen watermark confidences', np.mean(derived_unseen_watermark_confidences))
print('Unrelated unseen watermark confidences', np.mean(unrelated_unseen_watermark_confidences))

print('Derived unseen true confidences', np.mean(derived_unseen_true_confidences))
print('Unrelated unseen true confidences', np.mean(unrelated_unseen_true_confidences))

print('Derived unseen watermark accuracy', np.mean(derived_watermark_accs))
print('Unrelated unseen watermark accuracy', np.mean(unrelated_watermark_accs))

print('Watermark size', np.mean(watermark_sizes))
