from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve, auc


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import json


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def build_model():

    inp = keras.layers.Input(shape=(784,))
    l1 = keras.layers.Dense(512, activation = 'sigmoid', kernel_initializer = 'glorot_normal', name='input')(inp)
    l2 = keras.layers.Dense(256, activation = 'sigmoid', kernel_initializer = 'glorot_normal')(l1)
    z = keras.layers.Dense(128, activation = 'sigmoid', kernel_initializer = 'glorot_normal', name='latent')(l2)
    d1 = keras.layers.Dense(256, activation = 'sigmoid', kernel_initializer='glorot_normal')(z)
    d2 = keras.layers.Dense(512, activation = 'sigmoid', kernel_initializer='glorot_normal')(d1)
    #d2 = keras.layers.Dropout(0.25)(d2)
    d3 = keras.layers.Dense(784, activation = 'sigmoid', kernel_initializer='glorot_normal', name = 'output')(d2)

    model = keras.models.Model(inp, d3)
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model


def compute_auc(model, cat_name, data, checkpoint_path):

    normal_indx = [i for i, cat in enumerate(class_names) if cat == cat_name][0]

    inputs = data.test.images
    labels = data.test.labels
    outputs = model.predict(x = inputs)
    scores = K.eval(K.mean(K.square(inputs - outputs), axis=-1))
    labels_normal = [1 if label == normal_indx else 0 for label in labels]
    fpr, tpr, thresholds = roc_curve(labels_normal, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)

    np.savetxt(checkpoint_path + os.sep + "fpr.txt", fpr)
    np.savetxt(checkpoint_path + os.sep + "tpr.txt", tpr)
    np.savetxt(checkpoint_path + os.sep + "thresholds.txt", thresholds)
    np.savetxt(checkpoint_path + os.sep + "scores.txt", scores)

    f = open(checkpoint_path + os.sep + "AUC.txt", "a")
    f.write("AUC:{}".format(roc_auc))
    f.close()

def train(dataset, batch_size, epoch, cat_name, data):

    main_path = cat_name.replace(os.sep, '_') + '_AE'
    if not(os.path.isdir(main_path)):
        os.mkdir(main_path)

    model = build_model()

    for i in range(epoch):

        checkpoint_path = main_path + os.sep + str(i) + '.' + "weights.hdf5"

        cp_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                      save_weights_only = True,
                                                      verbose = 0,
                                                      monitor = 'val_loss',
                                                      save_best_only = True,
                                                      mode='min')

        out = model.fit(x = dataset, validation_split = 0.2, y = dataset, batch_size = batch_size, epochs = 1, callbacks = [cp_callback], verbose = 0)
        print("epoch:{} *** training loss:{} *** validation loss:{}".format(i, np.average(out.history['loss']), np.average(out.history['val_loss'])))
        f = open(main_path + os.sep + "log.txt", "a")
        f.write("epoch:{} *** training loss:{} *** validation loss:{}\n".format(i, np.average(out.history['loss']), np.average(out.history['val_loss'])))
        f.write("\n******************************\n")
        f.write(json.dumps(out.history))
        f.write("\n******************************\n")
        f.close()

    compute_auc(model, cat_name, data, main_path)



def train_categories(data, epoch, batch_size, classes):


    for cat, cat_name in enumerate(class_names):
        if (classes != -1) and (cat == classes):
            print("Training on {} started".format(cat_name))
            mask = data.train.labels == cat
            dataset = data.train.images[mask]
            print("Number of training samples: {}".format(len(dataset)))
            train(dataset, batch_size, epoch, cat_name, data)
        elif classes == -1:
            print("Training on {} started".format(cat_name))
            mask = data.train.labels == cat
            dataset = data.train.images[mask]
            print("Number of training samples: {}".format(len(dataset)))
            train(dataset, batch_size, epoch, cat_name, data)




def main(epoch, batch_size, gpu_id, data_path, classes):

    data = input_data.read_data_sets(data_path, source_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

    if gpu_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config = config)

    train_categories(data, epoch, batch_size, classes)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trains AE using adverserial objective function")
    parser.add_argument("-g", "--gpu_id", default = '-1', type = str, help="determines gpu id")
    parser.add_argument("-d", "--data_path", default = ".{}fashion_mnist".format(os.sep), help = 'path to dataset')
    parser.add_argument("-c", "--checkpoint_path", default = "large_latent{}weights.hdf5".format(os.sep), help = "the address in which the model is going to be saved.")
    parser.add_argument("-e", "--epoch", default = 700, type = int, help = "number of epochs")
    parser.add_argument("-b", "--batch_size", default = 256, type = int, help = "mini batch size")
    parser.add_argument("-l", "--classes", default = -1, type = int, help = "determines category on which you intend to train a model")

    args = parser.parse_args()
    main(args.epoch, args.batch_size, args.gpu_id, args.data_path, args.classes)
