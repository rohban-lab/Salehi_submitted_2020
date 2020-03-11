import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from train import autoencoder


def compute_auc(model, test_images, test_labels):
    # Computing reconstruction loss
    y_pred = model.predict(np.concatenate((test_images, test_images), axis=-1))
    diff = y_pred - test_images
    diff = np.sum(diff ** 2, axis=1)

    # Computing AUC
    print(diff)
    print(test_labels)
    fpr, tpr, thresholds = roc_curve(test_labels, diff, 1)
    AUC = auc(fpr, tpr)
    print('AUC: ' + str(AUC))

    # Plotting ROC
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title('ROC')
    plt.show()


def compute_f1(model, test_images, test_labels, validation):
    # Computing reconstruction loss
    y_pred = model.predict(np.concatenate((test_images, test_images), axis=-1))
    diff = y_pred - test_images
    diff = np.sum(diff ** 2, axis=1)

    # Computing F1 score
    tmin = min(diff) - 1
    tmax = max(diff) + 1
    f1 = 0
    best_threshold = 0
    for threshold in np.arange(tmin, tmax, 0.1):
        pred = [int(d > threshold) for d in diff]
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for i in range(len(pred)):
            if test_labels[i] == pred[i] == 1:
                true_positive += 1
            if pred[i] == 1 and test_labels[i] != pred[i]:
                false_positive += 1
            if test_labels[i] == pred[i] == 0:
                true_negative += 1
            if pred[i] == 0 and test_labels[i] != pred[i]:
                false_negative += 1

        if true_positive + false_positive == 0 or true_positive + false_negative == 0:
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision + recall:
            continue
        score = 2 * precision * recall / (precision + recall)
        if score > f1:
            f1 = score
            best_threshold = threshold

    if validation:
        return best_threshold
    else:
        print('F1: ' + str(f1))


if __name__ == '__main__':
    dataset = np.load('data/meta.npy')[0]
    test_images = np.load('data/test_images.npy')
    test_labels = np.load('data/test_labels.npy')
    model = autoencoder(3072, 0.1)
    model.load_weights('model/weights.hdf5')
    if dataset == 'fashion_mnist' or dataset == 'mnist':
        pass
    elif dataset == 'coil100':
        compute_auc(model, test_images, test_labels)
        compute_f1(model, test_images, test_labels, False)
