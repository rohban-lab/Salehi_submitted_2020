import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from train import autoencoder
from prepare import protocol1, protocol2


def prepare_pretrained_model(directory, *args):
    # Preparing the data for a pre-trained model
    dir_split = directory.split('/')
    if dir_split[2] == 'mnist_pretrained':
        if dir_split[3] == 'p1':
            protocol1('mnist', [int(dir_split[4])], float(args[0]))
        elif dir_split[3] == 'p2':
            protocol2('mnist', int(dir_split[4]))
    elif dir_split[2] == 'fashion_mnist_pretrained':
        protocol2('fashion_mnist', int(dir_split[3]))
    elif dir_split[2] == 'coil100_pretrained':
        if dir_split[3] == '1':
            anomaly_percentage = 0.5
        elif dir_split[3] == '4':
            anomaly_percentage = 0.25
        elif dir_split[3] == '7':
            anomaly_percentage = 0.15
        class_numbers = list(np.load(directory + 'class.npy'))
        protocol1('coil100', class_numbers, anomaly_percentage)


def compute_auc(model, test_images, test_labels, normal_class):
    # Computing reconstruction loss
    y_pred = model.predict(np.concatenate((test_images, test_images), axis=-1))
    diff = y_pred - test_images
    diff = np.sum(diff ** 2, axis=1)
    if normal_class == 1:
        diff = -diff

    # Computing AUC
    fpr, tpr, thresholds = roc_curve(test_labels, diff, 1)
    AUC = auc(fpr, tpr)
    print('AUC: ' + str(AUC))

    # Plotting ROC
    plt.plot(fpr, tpr)
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title('ROC')
    plt.show()


def get_f1(threshold, diff, labels, normal_class):
    if normal_class == 0:
        pred = [int(d > threshold) for d in diff]
    else:
        pred = [int(d < threshold) for d in diff]
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(pred)):
        if labels[i] == pred[i] == 1:
            true_positive += 1
        if pred[i] == 1 and labels[i] != pred[i]:
            false_positive += 1
        if labels[i] == pred[i] == 0:
            true_negative += 1
        if pred[i] == 0 and labels[i] != pred[i]:
            false_negative += 1

    if true_positive + false_positive == 0 or true_positive + false_negative == 0:
        return 0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def find_f1(model, test_images, test_labels, validation, *args):
    if validation:
        # Finding the best threshold using the validation set
        y_pred = model.predict(np.concatenate((args[0], args[0]), axis=-1))
        diff = y_pred - args[0]
        diff = np.sum(diff ** 2, axis=1)

        tmin = min(diff) - 1
        tmax = max(diff) + 1
        f1 = 0
        best_threshold = 0
        for threshold in np.arange(tmin, tmax, 0.1):
            score = get_f1(threshold, diff, args[1], 1)
            if score > f1:
                f1 = score
                best_threshold = threshold

        # Computing reconstruction loss
        y_pred = model.predict(np.concatenate((test_images, test_images), axis=-1))
        diff = y_pred - test_images
        diff = np.sum(diff ** 2, axis=1)

        # Compuring F1 score
        f1 = get_f1(best_threshold, diff, test_labels, 1)
        print('F1: ' + str(f1))
    else:
        # Computing reconstruction loss
        y_pred = model.predict(np.concatenate((test_images, test_images), axis=-1))
        diff = y_pred - test_images
        diff = np.sum(diff ** 2, axis=1)

        # Computing F1 score
        tmin = min(diff) - 1
        tmax = max(diff) + 1
        f1 = 0
        for threshold in np.arange(tmin, tmax, 0.1):
            score = get_f1(threshold, diff, test_labels, 0)
            if score > f1:
                f1 = score
        print('F1: ' + str(f1))


if __name__ == '__main__':
    args = sys.argv
    model_directory = 'model/'
    if len(args) > 1:
        model_directory = args[1]
        if len(args) == 2:
            prepare_pretrained_model(args[1])
        else:
            prepare_pretrained_model(args[1], args[2])

    # Loading the data
    dataset, protocol = np.load('data/meta.npy')
    test_images = np.load('data/test_images.npy')
    test_labels = np.load('data/test_labels.npy')

    # Loading the model
    model = autoencoder(test_images.shape[1], 0.1)
    model.load_weights(model_directory + 'weights.hdf5')

    # Computing AUC and F1 score
    if dataset == 'fashion_mnist' or dataset == 'mnist':
        if protocol == 'p1':
            validation_images = np.load('data/validation_images.npy')
            validation_labels = np.load('data/validation_labels.npy')
            find_f1(model, test_images, test_labels, True, validation_images, validation_labels)
            compute_auc(model, test_images, test_labels, 1)
        elif protocol == 'p2':
            compute_auc(model, test_images, test_labels, 0)
    elif dataset == 'coil100':
        find_f1(model, test_images, test_labels, False)
        compute_auc(model, test_images, test_labels, 0)
