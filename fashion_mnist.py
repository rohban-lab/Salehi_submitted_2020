from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras import backend as K


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return keras.layers.Lambda(func)


def find_delta(model, images, epsilon, learning_rate, steps, coef):

    latent_function = K.function([model.get_layer('input').get_input_at(0)],
                                 [model.get_layer('latent').get_output_at(0)])
    latent = keras.Input(shape=(128,))

    loss = coef*K.mean((model.get_layer('latent').get_output_at(0) - latent) ** 2, axis=-1) \
    + K.mean(K.square(model.get_layer('output').get_output_at(0) - images),axis=-1)

    loss_function = K.function([model.get_layer('input').get_input_at(0), latent], [loss])

    ce = K.gradients(loss, model.get_layer('input').get_input_at(0))

    gradient_function = K.function([model.get_layer('input').get_input_at(0), latent], ce)

    latent_images = latent_function([images])[0]

    delta = np.random.random((images.shape[0], 784)) * 2 * epsilon - epsilon

    for step in range(steps):
        attack_images = images + delta
        attack_images = np.clip(attack_images, 0, 1)

        loss_val = loss_function([attack_images, latent_images])[0]

        output = gradient_function([attack_images, latent_images])[0]

        delta = delta + learning_rate * np.sign(output)

        indices = np.nonzero(delta > epsilon)
        delta[indices[0], indices[1]] = epsilon

        indices = np.nonzero(delta < -epsilon)
        delta[indices[0], indices[1]] = -epsilon

    attack_images = images + delta
    attack_images = np.clip(attack_images, 0, 1)
    print(np.average(loss_val))
    return np.array(attack_images), np.array(images)


def attack_loss(coef, z1, zdelta):

    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1) + coef * K.mean(K.square(z1 - zdelta), axis=-1)

    return loss


def build_model(coef):

    inp = keras.layers.Input(shape=(1568,))
    main_inp = crop(1, 0, 784)(inp)
    aux_inp = crop(1, 784, 1568)(inp)
    noise = keras.layers.GaussianNoise(0, input_shape=(784,))
    l1 = keras.layers.Dense(512, activation = 'sigmoid', kernel_initializer = 'glorot_normal', name='input')
    l2 = keras.layers.Dense(256, activation = 'sigmoid', kernel_initializer = 'glorot_normal')
    z = keras.layers.Dense(128, activation = 'sigmoid', kernel_initializer = 'glorot_normal', name='latent')
    adelta1 = noise(main_inp)
    adelta2 = l1(adelta1)
    adelta3 = l2(adelta2)
    zdelta = z(adelta3)
    a1 = noise(aux_inp)
    a2 = l1(a1)
    a3 = l2(a2)
    z1 = z(a3)
    d1 = keras.layers.Dense(256, activation = 'sigmoid', kernel_initializer='glorot_normal')(zdelta)
    d2 = keras.layers.Dense(512, activation = 'sigmoid', kernel_initializer='glorot_normal')(d1)
    d3 = keras.layers.Dense(784, activation = 'sigmoid', kernel_initializer='glorot_normal', name = 'output')(d2)
    model = keras.models.Model(inp, d3)
    model.compile(optimizer = 'adam', loss = attack_loss(coef, z1, zdelta), metrics = [],)
    return model


def train(dataset, batch_size, coef, class_number, epoch, epsilon, steps):

    checkpoint_path = str(class_number) + os.sep + "weights.hdf5"

    if not(os.path.isdir(str(class_number))):
        os.mkdir(str(class_number))
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = build_model(coef)



    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    for i in range(epoch):
        print('step: {}'.format(i))
        attack_images, images = find_delta(model, dataset, epsilon, 2.5 * epsilon / steps, steps, coef)

        model.fit(x=np.concatenate((attack_images, images), axis=-1), y=images, batch_size=batch_size, epochs=1,
                  callbacks=[cp_callback], verbose=2)





def train_categories(data, epoch, batch_size, coef, epsilon, steps, classes):

    digitDict = {}

    for i in range(len(class_names)):
        mask = (data.train.labels == i)
        digitDict[i] = data.train.images[mask]

    if classes != -1:
        enu_classes = [class_names[classes]]
    else:
        enu_classes = class_names

    for cat, cat_name in enumerate(enu_classes):
        print("Training on {} started".format(cat_name))
        mask = data.train.labels == cat
        dataset = data.train.images[mask]
        train(dataset, batch_size, coef, cat, epoch, epsilon, steps)




def main(epoch, batch_size, coef, gpu_id, epsilon, steps, data_path, classes):

    data = input_data.read_data_sets(data_path, source_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

    if gpu_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config = config)

    train_categories(data, epoch, batch_size, coef, epsilon, steps, classes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trains AE using adverserial objective function")
    parser.add_argument("-g", "--gpu_id", default = '-1', type = str, help="determines gpu id")
    parser.add_argument("-d", "--data_path", default = ".{}fashion_mnist".format(os.sep), help = 'path to dataset')
    parser.add_argument("-c", "--checkpoint_path", default = "large_latent{}weights.hdf5".format(os.sep), help = "the address in which the model is going to be saved.")
    parser.add_argument("-e", "--epoch", default = 700, type = int, help = "number of epochs")
    parser.add_argument("-b", "--batch_size", default = 256, type = int, help = "mini batch size")
    parser.add_argument("-k", "--coef", default = 0.1, type = float, help = "setting coeficient in error function to control the effect of adverserial attack")
    parser.add_argument("-p", "--epsilon", default = 0.2, type = float, help = "epsilon")
    parser.add_argument("-s", "--steps", default = 40, type = int, help = "steps")
    parser.add_argument("-l", "--classes", default = -1, type = int, help = "determines category on which you intend to train a model")

    args = parser.parse_args()


    main(args.epoch, args.batch_size, args.coef, args.gpu_id, args.epsilon, args.steps, args.data_path, args.classes)
