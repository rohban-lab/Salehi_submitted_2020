import sys
import random
import numpy as np
import cv2
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


def protocol1(dataset, class_numbers, anomaly_percentage=0.5):
    # Loading and shuffling the dataset
    # Train and test splits are merged
    if dataset == 'coil100':
        dataset = np.array(list(tfds.as_numpy(tfds.load(name=dataset, split=['train'])[0])))
    else:
        dataset = np.array(list(tfds.as_numpy(tfds.load(name=dataset, split=['train+test'])[0])))
    random.shuffle(dataset)

    # Preparing the training data using 80% of the normal samples
    normal_samples = np.array(
        [cv2.resize(x['image'], (32, 32), interpolation=cv2.INTER_AREA).flatten() / 255.0 for x in dataset if
         x['label'] in class_numbers])
    abnormal_samples = np.array(
        [cv2.resize(x['image'], (32, 32), interpolation=cv2.INTER_AREA).flatten() / 255.0 for x in dataset if
         x['label'] not in class_numbers])
    train_images = normal_samples[:int(4 * len(normal_samples) / 5)]

    # Preparing the test data
    # A set of abnormal samples is prepared in the case a validation set is needed
    test_normal_samples = normal_samples[int(4 * len(normal_samples) / 5):]
    normal_count = len(test_normal_samples)
    abnormal_count = int(normal_count * anomaly_percentage / (1 - anomaly_percentage))

    if 2 * abnormal_count <= len(abnormal_samples):
        test_abnormal_samples = abnormal_samples[:abnormal_count]
        validation_abnormal_samples = abnormal_samples[abnormal_count:2 * abnormal_count]
    else:
        abnormal_count = int(len(abnormal_samples) / 2)
        normal_count = int(abnormal_count * (1 - anomaly_percentage) / anomaly_percentage)
        test_abnormal_samples = abnormal_samples[:abnormal_count]
        validation_abnormal_samples = abnormal_samples[abnormal_count:]
        test_normal_samples = test_normal_samples[:normal_count]

    test_images = np.concatenate((test_normal_samples, test_abnormal_samples))
    test_labels = np.concatenate((np.zeros(normal_count, dtype=int), np.ones(abnormal_count, dtype=int)))

    # Saving the data
    np.save('data/train_images', train_images)
    np.save('data/test_images', test_images)
    np.save('data/test_labels', test_labels)
    np.save('data/validation_abnormal_samples', validation_abnormal_samples)


def protocol2(dataset, class_number):
    # Loading the dataset and preparing the training data
    train, test = np.array(list(tfds.as_numpy(tfds.load(name=dataset, split=['train', 'test']))))
    train_images = np.array([x['image'].flatten() / 255.0 for x in train if x == class_number])

    # Preparing the test data
    test_images = np.array([x['image'].flatten() / 255.0 for x in train])
    test_labels = np.array([int(x['label'] != class_number) for x in train])

    # Saving the data
    np.save('data/train_images', train_images)
    np.save('data/test_images', test_images)
    np.save('data/test_labels', test_labels)


if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'mnist' or args[1] == 'fashion_mnist':
        if args[2] == 'p1':
            protocol1(args[1], [int(args[4])], float(args[3]))
        elif args[2] == 'p2':
            protocol2(args[1], int(args[3]))
    elif args[1] == 'coil100':
        protocol1(args[1], list(map(int, args[4:4 + int(args[3])])), float(args[2]))
    np.save('data/meta', np.array([args[1]]))
