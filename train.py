import os
import numpy as np
import keras
from keras import backend as K


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


def craft_adversarial_samples(model, images, epsilon, learning_rate, steps):
    # Normal images' latent
    latent_function = K.function([model.get_layer('input').get_input_at(0)],
                                 [model.get_layer('latent').get_output_at(0)])
    latent_images = latent_function([images])[0]

    # Latent loss
    latent = keras.Input(shape=(128,))
    loss = K.mean((model.get_layer('latent').get_output_at(0) - latent) ** 2, axis=-1)

    # Function to compute the gradient towards the input layer
    gradient = K.gradients(loss, model.get_layer('input').get_input_at(0))
    gradient_function = K.function([model.get_layer('input').get_input_at(0), latent], gradient)

    # Using PGD to craft the adversarial samples
    delta = np.random.uniform(-epsilon, epsilon, images.shape)
    advesarial_images = images + delta
    advesarial_images = np.clip(advesarial_images, 0, 1)
    for step in range(steps):
        # Updating delta
        grad = gradient_function([advesarial_images, latent_images])[0]
        delta = delta + learning_rate * np.sign(grad)

        # Projection step
        indices = np.nonzero(delta > epsilon)
        delta[indices[0], indices[1]] = epsilon
        indices = np.nonzero(delta < -epsilon)
        delta[indices[0], indices[1]] = -epsilon

        # Updating the adversarial samples
        advesarial_images = images + delta
        advesarial_images = np.clip(advesarial_images, 0, 1)

    return np.array(advesarial_images)


def AE_loss(gamma, norm_latent, adv_latent):

    def loss(y_true, y_pred):
        # AE loss which is the weighted sum of rec. loss and latent loss
        return K.mean(K.square(y_pred - y_true), axis=-1) + gamma * K.mean(K.square(norm_latent - adv_latent), axis=-1)

    return loss


def autoencoder(inp_size, gamma):
    # Encoder layers
    inp = keras.layers.Input(shape=(2 * inp_size,))
    adversarial_inp = crop(1, 0, inp_size)(inp)
    normal_inp = crop(1, inp_size, 2 * inp_size)(inp)
    e1 = keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal', name='input')
    e2 = keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal')
    latent = keras.layers.Dense(128, activation='sigmoid', kernel_initializer='glorot_normal', name='latent')

    # Encoding the adversarial  input
    adv1 = e1(adversarial_inp)
    adv2 = e2(adv1)
    adv_latent = latent(adv2)

    # Encoding the normal input
    norm1 = e1(normal_inp)
    norm2 = e2(norm1)
    norm_latent = latent(norm2)

    # Decoding the adversarial input
    d1 = keras.layers.Dense(256, activation='sigmoid', kernel_initializer='glorot_normal')(adv_latent)
    d2 = keras.layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_normal')(d1)
    out = keras.layers.Dense(inp_size, activation='sigmoid', kernel_initializer='glorot_normal', name='output')(d2)

    # Building the model
    model = keras.models.Model(inp, out)
    model.compile(optimizer='adam',
                  loss=AE_loss(gamma, norm_latent, adv_latent),
                  )

    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint_path = 'model/weights.hdf5'

    # Hyperparameters
    gamma = 0.1
    dataset = np.load('data/meta.npy')[0]
    if dataset == 'mnist':
        epsilon = 0.2
        num_steps = 1600
        batch_size = 256
        pgd_steps = 40
    elif dataset == 'fashion_mnist':
        epsilon = 0.05
        num_steps = 8000
        batch_size = 256
        pgd_steps = 40
    elif dataset == 'coil100':
        epsilon = 0.05
        num_steps = 1000
        batch_size = 16
        pgd_steps = 20

    # Loading training data
    train_images = np.load('data/train_images.npy')

    # Building the model
    model = autoencoder(train_images.shape[1], gamma)

    # Training the model
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
    for i in range(num_steps):
        print('step ' + str(i+1) + '/' + str(num_steps))
        # Adversarial sample crafting
        adversarial_samples = craft_adversarial_samples(model, train_images, epsilon, 2.5 * epsilon / pgd_steps,
                                                        pgd_steps)
        # Autoencoder adversarial training
        model.fit(x=np.concatenate((adversarial_samples, train_images), axis=-1), y=train_images, batch_size=batch_size,
                  epochs=1, callbacks=[cp_callback], verbose=2)
