# import innvestigate
# import innvestigate.utils
import os
import random
# import math

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
from keras import backend as K

class_number = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# print(K.tensorflow_backend._get_available_gpus())
# print(device_lib.list_local_devices())
mnist = input_data.read_data_sets("fashion/", source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=False)
checkpoint_path = str(class_number) + "/weights.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)


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


def find_delta(model, images, epsilon, learning_rate, steps):

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
        # print(np.average(loss_val))

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


def attack_loss(coef, z1, zdelta):

    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1) + coef * K.mean(K.square(z1 - zdelta), axis=-1)

    return loss


digitDict = {}

for i in range(10):
    mask = (mnist.train.labels == i)
    digitDict[i] = mnist.train.images[mask]
mask = mnist.train.labels == class_number
dataset = mnist.train.images[mask]
# random.shuffle(dataset)
# cv = dataset[4800:5389]
# dataset = dataset[:4800]
# dataset2 = np.add(dataset, np.full(dataset.shape, 0.2))
show_image(dataset[0])

learning_rate = 0.001
num_steps = 700
batch_size = 256
coef = 0.1
drop_rate = 0.05

display_step = 1000
display_step_1 = 5000
examples_to_show = 10

# def create_model():
#     model = keras.models.Sequential([
#         keras.layers.Input(shape=(784,)),
#         keras.layers.GaussianNoise(0, input_shape=(784,)),
#         keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(128, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(784, activation='sigmoid', kernel_initializer='glorot_normal')
#     ])
#
#     # latent_function = K.function([model.layers[0].input], [model.layers[3].output])
#
#     model.compile(optimizer='adam',
#                   loss='mean_squared_error',
#                   metrics=['accuracy'])
#
#     return model

# # inp = keras.layers.Input(shape=(1568,))
# # main_inp = crop(1, 0, 784)(inp)
# # aux_inp = crop(1, 784, 1568)(inp)
# main_inp = keras.layers.Input(shape=(784,))
# d0 = keras.layers.Dropout(drop_rate, name='input')
# l1 = keras.layers.Dense(512, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(3))
# b1 = keras.layers.BatchNormalization()
# ac1 = keras.layers.Activation('relu')
# d1 = keras.layers.Dropout(drop_rate)
# l2 = keras.layers.Dense(256, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(3))
# b2 = keras.layers.BatchNormalization()
# ac2 = keras.layers.Activation('relu')
# d2 = keras.layers.Dropout(drop_rate)
# z = keras.layers.Dense(128, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(3))
# bz = keras.layers.BatchNormalization()
# acz = keras.layers.Activation('relu', name='latent')
# dz = keras.layers.Dropout(drop_rate)
#
# # a1 = d0(aux_inp)
# # a1 = l1(a1)
# # a1 = b1(a1)
# # a1 = ac1(a1)
# # a1 = d1(a1)
# # a2 = l2(a1)
# # a2 = b2(a2)
# # a2 = ac2(a2)
# # a2 = d2(a2)
# # z1 = z(a2)
# # z1 = bz(z1)
# # z1 = acz(z1)
#
# adelta1 = d0(main_inp)
# adelta1 = l1(adelta1)
# adelta1 = b1(adelta1)
# adelta1 = ac1(adelta1)
# adelta1 = d1(adelta1)
# adelta2 = l2(adelta1)
# adelta2 = b2(adelta2)
# adelta2 = ac2(adelta2)
# adelta2 = d2(adelta2)
# zdelta = z(adelta2)
# zdelta = bz(zdelta)
# zdelta = acz(zdelta)
# fzdelta = dz(zdelta)
#
# d1 = keras.layers.Dense(256, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(3))(fzdelta)
# d1 = keras.layers.BatchNormalization()(d1)
# d1 = keras.layers.Activation('relu')(d1)
# d1 = keras.layers.Dropout(drop_rate)(d1)
# d2 = keras.layers.Dense(512, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(3))(d1)
# d2 = keras.layers.BatchNormalization()(d2)
# d2 = keras.layers.Activation('relu')(d2)
# d2 = keras.layers.Dropout(drop_rate)(d2)
# d3 = keras.layers.Dense(784, kernel_initializer='glorot_normal', kernel_constraint=keras.constraints.max_norm(3))(d2)
# d3 = keras.layers.BatchNormalization()(d3)
# d3 = keras.layers.Activation('relu')(d3)
# model = keras.models.Model(main_inp, d3)

inp = keras.layers.Input(shape=(1568,))
main_inp = crop(1, 0, 784)(inp)
aux_inp = crop(1, 784, 1568)(inp)
noise = keras.layers.GaussianNoise(0, input_shape=(784,))
l1 = keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal', name='input')
l2 = keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal')
z = keras.layers.Dense(128, activation='sigmoid', kernel_initializer='glorot_normal', name='latent')
adelta1 = noise(main_inp)
adelta2 = l1(adelta1)
adelta3 = l2(adelta2)
zdelta = z(adelta3)
a1 = noise(aux_inp)
a2 = l1(a1)
a3 = l2(a2)
z1 = z(a3)
# encoder = keras.models.Sequential([inp, l1, l2, z])(z)
d1 = keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal')(zdelta)
d2 = keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal')(d1)
d3 = keras.layers.Dense(784, activation='sigmoid', kernel_initializer='glorot_normal', name='output')(d2)
model = keras.models.Model(inp, d3)

# inp = keras.layers.Input(shape=(1568,))
# main_inp = crop(1, 0, 784)(inp)
# aux_inp = crop(1, 784, 1568)(inp)
# noise = keras.layers.GaussianNoise(0.1, input_shape=(784,))
# main_inp = noise(main_inp)
# aux_inp = noise(aux_inp)
# r1 = keras.layers.Reshape((28, 28, 1))
# main_inp = r1(main_inp)
# aux_inp = r1(aux_inp)
#
# l1 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), kernel_initializer='glorot_normal', padding='valid',
#                          name='input')
# b1 = keras.layers.BatchNormalization()
# ac1 = keras.layers.Activation('relu')
# l2 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), kernel_initializer='glorot_normal', padding='valid')
# b2 = keras.layers.BatchNormalization()
# ac2 = keras.layers.Activation('relu')
# l3 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), kernel_initializer='glorot_normal', padding='valid')
# b3 = keras.layers.BatchNormalization()
# ac3 = keras.layers.Activation('relu')
# z = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), kernel_initializer='glorot_normal', padding='valid')
# bz = keras.layers.BatchNormalization()
# acz = keras.layers.Activation('relu')
# f = keras.layers.Flatten(name='latent')
# r2 = keras.layers.Reshape((12, 12, 256))
#
# a1 = l1(aux_inp)
# a1 = b1(a1)
# a1 = ac1(a1)
# a2 = l2(a1)
# a2 = b2(a2)
# a2 = ac2(a2)
# a3 = l3(a2)
# a3 = b3(a3)
# a3 = ac3(a3)
# z1 = z(a3)
# z1 = bz(z1)
# z1 = acz(z1)
# z1 = f(z1)
#
# adelta1 = l1(main_inp)
# adelta1 = b1(adelta1)
# adelta1 = ac1(adelta1)
# adelta2 = l2(adelta1)
# adelta2 = b2(adelta2)
# adelta2 = ac2(adelta2)
# adelta3 = l3(adelta2)
# adelta3 = b3(adelta3)
# adelta3 = ac3(adelta3)
# zdelta = z(adelta3)
# zdelta = bz(zdelta)
# zdelta = acz(zdelta)
# zdelta = f(zdelta)
# fzdelta = r2(zdelta)
#
# d1 = keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='valid',
#                                   kernel_initializer='glorot_normal')(fzdelta)
# d1 = keras.layers.BatchNormalization()(d1)
# d1 = keras.layers.Activation('relu')(d1)
# d2 = keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), padding='valid',
#                                   kernel_initializer='glorot_normal')(d1)
# d2 = keras.layers.BatchNormalization()(d2)
# d2 = keras.layers.Activation('relu')(d2)
# d3 = keras.layers.Conv2DTranspose(filters=32, kernel_size=(5, 5), padding='valid',
#                                   kernel_initializer='glorot_normal')(d2)
# d3 = keras.layers.BatchNormalization()(d3)
# d3 = keras.layers.Activation('relu')(d3)
# d4 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), padding='valid',
#                                   kernel_initializer='glorot_normal')(d3)
# d4 = keras.layers.BatchNormalization()(d4)
# d4 = keras.layers.Activation('relu')(d4)
# d4 = keras.layers.Reshape((784,))(d4)
# model = keras.models.Model(inp, d4)

# model.compile(optimizer='adam',
#                   loss=attack_loss(coef, input_layer),
#                   metrics=['accuracy'])
# model = create_model()
# model.summary()
# model = keras.models.Sequential([
#         keras.layers.GaussianNoise(0, input_shape=(784,)),
#         keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(128, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal'),
#         keras.layers.Dense(784, activation='sigmoid', kernel_initializer='glorot_normal')
#     ])

# model.compile(optimizer='adam',
#                 loss=attack_loss(coef, z1, zdelta),
#                 metrics=['accuracy'],
#               )
model.compile(optimizer='adam',
              loss=attack_loss(coef, z1, zdelta),
              metrics=[],
              )


# model.load_weights(str(class_number) + '/weights.hdf5')

# latent_function = K.function([model.layers[0].input], [model.layers[3].output])
# print(latent_function([dataset])[0])
# print('dfv')

def draw_pca():
    latent_function = K.function([model.get_layer('input').get_input_at(0)],
                                 [model.get_layer('latent').get_output_at(0)])
    latent8 = latent_function([digitDict[8]])[0]
    latent3 = latent_function([digitDict[3]])[0]
    latent1 = latent_function([digitDict[1]])[0]
    latent = np.concatenate((latent8[0:500], latent3[0:500], latent1[0:500]), 0)
    print(np.mean(np.mean(latent8, axis=0)))
    print(np.mean(np.std(latent8, axis=0)))
    latent = latent - np.mean(latent, axis=0)
    latent /= np.std(latent, axis=0)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(latent)
    plt.scatter(*zip(*principalComponents),
                c=['r' for i in range(500)] + ['b' for i in range(500)] + ['k' for i in range(500)])
    # plt.savefig('plot3.jpg')
    plt.show()


# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
#
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

for i in range(num_steps):
    print('step:')
    print(i)
    attack_images, images = find_delta(model, dataset, 0.2, 2.5 * 0.2 / 40, 40)
    # attack_images2, images2 = find_delta(model, dataset[2500:], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images3, images3 = find_delta(model, dataset[1000:1500], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images4, images4 = find_delta(model, dataset[1500:2000], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images5, images5 = find_delta(model, dataset[2000:2500], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images6, images6 = find_delta(model, dataset[2500:3000], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images7, images7 = find_delta(model, dataset[3000:3500], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images8, images8 = find_delta(model, dataset[3500:4000], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images9, images9 = find_delta(model, dataset[4000:4500], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images10, images10 = find_delta(model, dataset[4500:], 0.0313, 2.5 * 0.0313 / 10, 10)
    # attack_images = np.concatenate((attack_images1, attack_images2), axis=0)
    # images = np.concatenate((images1, images2), axis=0)
    # if i % 5 == 4:
    #     v = 2
    # else:
    #     v = 0
    model.fit(x=np.concatenate((attack_images, images), axis=-1), y=images, batch_size=batch_size, epochs=1,
              callbacks=[cp_callback], verbose=2)
    # if i%10 == 9:
    #     draw_pca()

# model.fit(x=np.concatenate((np.array(dataset2), np.array(dataset2)), axis=-1), y=dataset, batch_size=batch_size, epochs=3000,callbacks=[cp_callback], verbose=2)
# # num_steps = 2
# draw_pca()
# for i in range(num_steps):
#     print(i)
#     # data = dataset[np.random.choice(len(digitDict[8]), 100, replace=False)]
#     attack_images1, images1 = find_delta(model, dataset[:700], 0.3, 2, 20)
#     attack_images2, images2 = find_delta(model, dataset[700:1400], 0.3, 2, 20)
#     attack_images3, images3 = find_delta(model, dataset[1400:2100], 0.3, 2, 20)
#     attack_images4, images4 = find_delta(model, dataset[2100:2800], 0.3, 2, 20)
#     attack_images5, images5 = find_delta(model, dataset[2800:3500], 0.3, 2, 20)
#     attack_images6, images6 = find_delta(model, dataset[3500:4200], 0.3, 2, 20)
#     attack_images7, images7 = find_delta(model, dataset[4200:4800], 0.3, 2, 20)
#     attack_images8, images8 = find_delta(model, dataset[4800:], 0.3, 2, 20)
#     attack_images = np.concatenate((attack_images1, attack_images2, attack_images3, attack_images4, attack_images5,
#                                     attack_images6, attack_images7, attack_images8), axis=0)
#     images = np.concatenate((images1, images2, images3, images4, images5, images6, images7, images8), axis=0)
#     if i % 5 == 4:
#         v = 2
#     else:
#         v = 0
#     model.fit(x=np.concatenate((attack_images, images), axis=-1), y=images, batch_size=batch_size, epochs=1,
#               callbacks=[cp_callback], verbose=v)
#     # if i%10 == 9:
#     #     draw_pca()
# draw_pca()

# def latent_loss(x):
#     def loss(delta):
#         latent_function = K.function([model.get_layer('input').get_input_at(0)], [model.get_layer('latent').get_output_at(0)])
#         return K.mean(K.square(latent_function(x+delta) - latent_function(x)), axis=-1)
#     return loss
# show_image(digitDict[1][0])
# res1 = None
# im = np.concatenate((np.array(digitDict[1][0]), np.array(digitDict[1][0])), axis=-1).reshape(1, 1568)
# print(im.shape)
# for i in range(100):
#     res1 = model.predict(x = im)[0].reshape(1, 784)
#     im = np.concatenate((np.array(res1), np.array(res1)), axis=-1)
#     show_image(res1)
# show_image(res1)

# show_image(dataset[1])
# input, output = find_delta(model, dataset[:5], 0.05, 1, 1000)
# print(model.evaluate(x=input, y=output, batch_size=batch_size))
# print(model.metrics_names)
# res1 = model.predict(x=input)[1].reshape(1, 784)
# show_image(input[1, :784].reshape(1, 784))
# show_image(res1)
#
# input, output = find_delta(model, cv[:5], 0.04, 1, 5000)
# print(model.evaluate(x=input, y=output, batch_size=batch_size))
# print(model.metrics_names)
# res1 = model.predict(x=input)[1].reshape(1, 784)
# show_image(input[1, :784].reshape(1, 784))
# show_image(res1)
#
# input, output = find_delta(model, cv[:5], 0.04, 2, 10000)
# print(model.evaluate(x=input, y=output, batch_size=batch_size))
# print(model.metrics_names)
# res1 = model.predict(x=input)[1].reshape(1, 784)
# show_image(input[1, :784].reshape(1, 784))
# show_image(res1)

# input, output = find_delta(model, cv[:5], 0.04, 2, 3000)
# print(model.evaluate(x=input, y=output, batch_size=batch_size))
# print(model.metrics_names)
# res1 = model.predict(x=input)[1].reshape(1, 784)
# show_image(input[1, :784].reshape(1, 784))
# show_image(res1)
# dataset2 = np.concatenate((np.array(dataset2), np.array(dataset2)), axis=-1)
# model.fit(x=dataset2, y=dataset, batch_size=batch_size, epochs=3000, callbacks=[cp_callback])

# # model.load_weights(checkpoint_path)
# # show_image(digitDict[3][5])
# # p = model.predict(digitDict[3])[5]
# # show_image(p)
# # show_image(model.predict(np.array([p])))
# # model.fit(np.array([np.concatenate((np.array(digitDict[3][5]), np.array(digitDict[3][5])))]), np.array(digitDict[3][5]).reshape(1, 784))
# # model.predict([np.array(digitDict[3][5]).reshape(1, 784), np.array(digitDict[3][5]).reshape(1, 784)] )
# analyzer = innvestigate.analyzer.LRPEpsilon(model, input_layer_rule=(0, 1), neuron_selection_mode='all', allow_lambda_layers=True)
# # print(np.array([digitDict[3][5], digitDict[3][5]]).shape)
# a = analyzer.analyze(np.array([np.concatenate((np.array(digitDict[8][7]), np.array(digitDict[8][7])))]))
# # print(np.ravel_multi_index((0, 11, 10, 0), a.shape))
# # print(np.array(a).shape)
# a = a[:, :784]
# a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
# a /= np.max(np.abs(a))
# show_image2(a)

# threshold = 60
# class_num = 4
# print("feed number 1 : ")
# y_pred = model.predict(digitDict[class_num])
# show_image(y_pred[0])
# diff = y_pred - digitDict[class_num]
# diff = np.sum(diff**2,axis=1)
# anomal_per = (diff > threshold).sum() / diff.shape[0]
# print(anomal_per)

# for i in range(10):
#     mask = (mnist.test.labels == i)
#     digitDict[i] = mnist.test.images[mask]
# mask = mnist.test.labels == class_number
# dataset = mnist.test.images[mask]
#
# x = []
# y = []
# norm_num = len(digitDict[class_number])
# anom_num = len(digitDict[0]) + len(digitDict[1]) + len(digitDict[2]) + len(digitDict[3]) + len(digitDict[4]) + len(
#     digitDict[5]) + len(digitDict[6]) + len(digitDict[7]) + len(digitDict[8]) + len(digitDict[9]) - len(
#     digitDict[class_number])
# auc = 0
# for threshold in range(0, 100, 5):
#     print(threshold)
#     truep = 0
#     falsep = 0
#     for i in range(10):
#         y_pred = model.predict(np.concatenate((digitDict[i], digitDict[i]), axis=-1))
#         diff = y_pred - digitDict[i]
#         diff = np.sum(diff ** 2, axis=1)
#         anomal_per = (diff > threshold).sum()
#         if i != class_number:
#             truep += anomal_per
#         else:
#             falsep += anomal_per
#     x.append(falsep / norm_num)
#     y.append(truep / anom_num)
#     if threshold != 0:
#         auc += (- x[int(threshold / 5)] + x[int(threshold / 5) - 1]) * (
#                 y[int(threshold / 5)] + y[int(threshold / 5) - 1]) / 2
# print(auc)
# plt.plot(x, y, '-ok')
# plt.title('ROC  k = 0.1, class = ' + str(class_number))
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.text(0.5, 0.5, 'AUC: ' + str(auc))
# plt.savefig('adv-class' + str(class_number))

# inp = digitDict[8] + np.random.normal(0, 0.01, digitDict[8].shape)
# inp = inp + 0.1
# print(inp[0])

# zaribe 7 va epsilon 0.1