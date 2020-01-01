import random
from timeit import default_timer as timer
import sklearn
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K
import cv2
import os
import argparse





def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            if end - start == 1:
                return x[:, start]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return keras.layers.Lambda(func)


def find_delta(model, images, epsilon, learning_rate, steps, coef):
    loss = K.square(model.get_layer('output').get_output_at(0) - model.get_layer('input').get_input_at(1)) \
           + coef * K.square(model.get_layer('latent').get_output_at(0) - model.get_layer('latent').get_output_at(1))
    loss = K.mean(loss, axis=(-1, -2, -3))

    loss_function = K.function([model.input], [loss])

    ce = K.gradients(loss, model.get_layer('input').get_input_at(1))

    gradient_function = K.function([model.input], ce)

    delta = np.random.random((images.shape[0], 61, 61,1)) * 2 * epsilon - epsilon

    inp = np.ndarray((images.shape[0], 2, 61, 61,1))

    for i in range(steps):
        print("delta step: ",i)

        attack_images = images + delta

        attack_images = np.clip(attack_images,-1,1)

        inp[:, 0, :, :,:] = attack_images
        inp[:, 1, :, :,:] = images

        loss_val = loss_function([inp])[0]
        print("delta loss: ", np.mean(loss_val))

        output = gradient_function([inp])[0]

        delta = delta + learning_rate * np.sign(output)

        indices = np.nonzero(delta > epsilon)
        delta[indices[0], indices[1],indices[2],indices[3]] = epsilon

        indices = np.nonzero(delta < -epsilon)
        delta[indices[0], indices[1],indices[2],indices[3]] = -epsilon

    attack_images = images + delta
    attack_images = np.clip(attack_images, -1, 1)

    inp[:, 0] = attack_images
    inp[:, 1] = images

    loss_val = loss_function([inp])[0]
    print("delta loss: ", np.mean(loss_val))

    return inp


def show_image(image):
    first_image = image
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def show_image2(image):
    first_image = image
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap="seismic", clim=(-1, 1))
    plt.show()


def attack_loss(coef, adelta4, a4):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=(-1,-2,-3))+ coef * K.mean(K.square(adelta4 - a4), axis=(-1,-2,-3))

    return loss


def show_image(input):
    n = 30
    canvas_orig = np.empty((61 * n, 122 * n))
    canvas_recon = np.empty((28 * n, 28 * n))

    for i in range(n):
        batch = input[i*n:(i+1)*n]

        g = model.predict(batch)

        # Display original images
        for j in range(2 * n):
            # Draw the original digits
            if j % 2 == 0:
                canvas_orig[i * 61:(i + 1) * 61, j * 61:(j + 1) * 61] = \
                   input[i * n + j // 2,0].reshape([61, 61])
            else:
                canvas_orig[i * 61:(i + 1) * 61, j * 61:(j + 1) * 61] = \
                    g[(j // 2)].reshape([61, 61])
    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()



def draw_pca():
    latent_function = K.function([model.get_layer('input').get_input_at(0)],
                                 [model.get_layer('latent').get_output_at(0)])
    latent8 = latent_function([np.reshape(np.array(digitDict[8]), (digitDict[8].shape[0], 28, 28, 1))])[0]
    latent3 = latent_function([np.reshape(np.array(digitDict[2]), (digitDict[2].shape[0], 28, 28, 1))])[0]
    latent1 = latent_function([np.reshape(np.array(digitDict[1]), (digitDict[1].shape[0], 28, 28, 1))])[0]
    latent = np.concatenate((latent8[0:500], latent3[0:500], latent1[0:500]), 0)
    print(np.mean(np.mean(latent8, axis=0)))
    print(np.mean(np.std(latent8, axis=0)))
    latent = latent - np.mean(latent, axis=0)
    latent /= np.std(latent, axis=0)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(latent)
    plt.scatter(*zip(*principalComponents),
                c=['r' for i in range(500)] + ['b' for i in range(500)] + ['k' for i in range(500)])
    plt.show()



def build_model(kernel_size, stride, coef):

    inp = keras.layers.Input(shape=(2, 61, 61, 1))

    noise_inp = crop(1,0,1)(inp)
    main_inp = crop(1,1,2)(inp)

    l1 = keras.layers.Conv2D(filters=64, strides=stride, kernel_size=kernel_size, kernel_initializer='glorot_normal',
                             padding='valid', name='input')
    ac1 = keras.layers.LeakyReLU(alpha=0.2)

    l2 = keras.layers.Conv2D(filters=128, kernel_size=kernel_size, kernel_initializer='glorot_normal', padding='valid',
                             strides=stride)
    b2 = keras.layers.BatchNormalization(momentum=0.1)
    ac2 = keras.layers.LeakyReLU(alpha=0.2)

    l3 = keras.layers.Conv2D(filters=256, kernel_size=kernel_size, kernel_initializer='glorot_normal', padding='valid',
                             strides=stride)
    b3 = keras.layers.BatchNormalization(momentum=0.1)
    ac3 = keras.layers.LeakyReLU(alpha=0.2)

    l4 = keras.layers.Conv2D(filters=16, kernel_size=kernel_size, kernel_initializer='glorot_normal', padding='valid',
                             strides=stride)
    ac4 = keras.layers.LeakyReLU(alpha=0.2,name='latent')

    z1 = keras.layers.Conv2DTranspose(filters=256, kernel_size=kernel_size, kernel_initializer='glorot_normal',
                                      padding='valid', strides=stride)
    bz1 = keras.layers.BatchNormalization(momentum=0.1)
    acz1 = keras.layers.Activation('relu')

    z2 = keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, kernel_initializer='glorot_normal',
                                      padding='valid', strides=stride)
    bz2 = keras.layers.BatchNormalization(momentum=0.1)
    acz2 = keras.layers.Activation('relu')

    z3 = keras.layers.Conv2DTranspose(filters=64, kernel_size=kernel_size, kernel_initializer='glorot_normal',
                                      padding='valid', strides=stride)
    bz3 = keras.layers.BatchNormalization(momentum=0.1)
    dp3 = keras.layers.Dropout(0.5)
    acz3 = keras.layers.Activation('relu')

    z4 = keras.layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, kernel_initializer='glorot_normal',
                                      padding='valid', strides=stride,activation='tanh',name='output')

    a1 = l1(main_inp)
    a1 = ac1(a1)

    a2 = l2(a1)
    a2 = b2(a2)
    a2 = ac2(a2)

    a3 = l3(a2)
    a3 = b3(a3)
    a3 = ac3(a3)

    a4 = l4(a3)
    a4 = ac4(a4)

    adelta1 = l1(noise_inp)
    adelta1 = ac1(adelta1)

    adelta2 = l2(adelta1)
    adelta2 = b2(adelta2)
    adelta2 = ac2(adelta2)

    adelta3 = l3(adelta2)
    adelta3 = b3(adelta3)
    adelta3 = ac3(adelta3)

    adelta4 = l4(adelta3)
    adelta4 = ac4(adelta4)

    z = z1(adelta4)
    z = bz1(z)
    z = acz1(z)

    z = z2(z)
    z = bz2(z)
    z = acz2(z)

    z = z3(z)
    z = bz3(z)
    z = dp3(z)
    z = acz3(z)

    z = z4(z)

    model = keras.models.Model(inp, z)
    model.compile(optimizer='adam', loss=attack_loss(coef,adelta4,a4),)

    return model




def main(gpu_id, data_path, checkpoint_path, class_number, learning_rate,
         num_steps, batch_size, coef, kernel_size, stride):

    if gpu_id != -1:

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    else:
        pass

    mnist = input_data.read_data_sets(data_path, one_hot=False)
    digitDict = {}

    model = build_model(kernel_size, stride, coef)

    # What is it doing?
    for i in range(10):
        mask = (mnist.train.labels == i)
        digitDict[i] = mnist.train.images[mask]
    mask = mnist.train.labels == class_number
    dataset = mnist.train.images[mask]
    random.shuffle(dataset)


    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, period=20)
    resized_dataset = np.ndarray((5389,61,61,1))

    # What this assignment is doing?
    dataset = dataset
    # Why again another loop over dataset?
    for i in range(5389):
        x = np.reshape(dataset[i],(28,28))
        resized_dataset[i,:,:,0] = cv2.resize(x,(61,61),interpolation=cv2.INTER_AREA)

    resized_dataset = resized_dataset/0.5 - 1


    for i in range(1000):
        input = np.ndarray((5389,2,61,61,1))

        data = resized_dataset[0:3000]
        input[0:3000] = find_delta(model, data.reshape((3000,61,61,1)), 0.2, learning_rate, 40, coef)

        data = resized_dataset[3000:5389]
        input[3000:5389] = find_delta(model, data.reshape((2389,61,61,1)), 0.2, learning_rate, 40, coef)

        model.fit(x=input, y=resized_dataset, batch_size=batch_size, epochs=1, callbacks=[cp_callback], verbose=1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trains AE using adverserial objective function")
    parser.add_argument("-g", "--gpu_id", default = -1, type = str, help="determines gpu id")
    parser.add_argument("-d", "--data_path", default = "MNIST_data{}".format(os.sep), help = 'path to dataset')
    parser.add_argument("-c", "--checkpoint_path", default = "large_latent{}weights.hdf5".format(os.sep), help = "the address in which the model is going to be saved.")
    parser.add_argument("-n", "--class_number", default = 8, type = int, help = "choose normal class")
    parser.add_argument("-l", "--learning_rate", default = 0.005, type = float, help = "learning rate")
    parser.add_argument("-s", "--num_steps", default = 100, type = int, help = "number of epochs")
    parser.add_argument("-b", "--batch_size", default = 256, type = int, help = "mini batch size")
    parser.add_argument("-k", "--coef", default = 0.1, type = float, help = "setting coeficient in error function to control the effect of adverserial attack")
    parser.add_argument("-e", "--kernel_size", default = 5, type = int, help = "kernel size")
    parser.add_argument("-t", "--stride", default = 2, type = int, help = "kernel stride")
    args = parser.parse_args()

    main(args.gpu_id, args.data_path, args.checkpoint_path,
        args.class_number, args.learning_rate, args.num_steps,
        args.batch_size, args.coef, args.kernel_size, args.stride)
