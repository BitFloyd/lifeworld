import keras.backend as K
import numpy as np
from keras import objectives
from keras.applications.vgg16 import VGG16
from keras.layers import Activation, GaussianNoise
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, merge, Dropout
from keras.layers import LSTM, concatenate, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from theano.tensor import set_subtensor
from theano.tensor.nnet.neighbours import images2neibs

alpha = 0.5
beta = 1 - alpha


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def weights_kick(model, kick=0.1):
    weights = model.get_weights()

    for j in range(0, len(weights)):
        noise = kick * np.random.normal(loc=weights[j].mean(), scale=weights[j].std(), size=weights[j].shape)
        weights[j] = weights[j] + noise

    model.set_weights(weights)


def loss_DSSIM_theano(y_true, y_pred):
    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
    # and cannot be used for learning
    y_true = y_true.dimshuffle([0, 3, 1, 2])
    y_pred = y_pred.dimshuffle([0, 3, 1, 2])
    patches_true = images2neibs(y_true, [4, 4])
    patches_pred = images2neibs(y_pred, [4, 4])

    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)

    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    std_true = K.sqrt(var_true + K.epsilon())
    std_pred = K.sqrt(var_pred + K.epsilon())

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero

    return (alpha * K.mean((1.0 - ssim) / 2.0) + beta * K.mean(K.square(y_pred - y_true), axis=-1))


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def vgg_model(shape, filter_list, maxpool=False, op_only_middle=True, highcap=True):
    input_img = Input(shape=shape)

    x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
    if (highcap):
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    if (op_only_middle):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)

    if (highcap):
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    decoded = Convolution2D(shape[2], 3, 3, activation='relu', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer=sgd, loss=loss_DSSIM_theano)
    return autoencoder


def vgg_model_non_linear(shape, maxpool=False, op_only_middle=True, highcap=True):
    input_img = Input(shape=shape)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
    if (highcap):
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    # Dense Layer with sigmoid ACTIVATION
    if (op_only_middle):
        x = Flatten()(x)
        x = Dense(3072, activation='sigmoid')(x)
        x = Reshape((32, 32, 3))(x)
    else:
        x = Flatten()(x)
        x = Dense(3072 * 4, activation='sigmoid')(x)
        x = Reshape((64, 64, 3))(x)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)

    if (highcap):
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)

    decoded = Convolution2D(shape[2], 3, 3, activation='relu', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer=sgd, loss=loss_DSSIM_theano)

    return autoencoder


def feature_comparison_model(shape, filter_list, maxpool=True, highcap=False, op_only_middle=False):
    input_img = Input(shape=shape)
    feat_extract_model = VGG16(include_top=False, weights='imagenet', input_shape=shape)

    x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
    if (highcap):
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    if (op_only_middle):
        x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[3], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[2], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (highcap):
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[1], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)

    if (highcap):
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
        x = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same')(x)
    if (maxpool):
        x = UpSampling2D((2, 2))(x)

    decoded = Convolution2D(shape[2], 3, 3, activation='relu', border_mode='same')(x)

    for i in range(0, len(feat_extract_model.layers)):
        feat_extract_model.layers[i].trainable = False

    autoencoder = Model(input_img, decoded)
    full_model = Sequential()

    for i in range(0, len(autoencoder.layers)):
        full_model.add(autoencoder.layers[i])

    for i in range(1, len(feat_extract_model.layers)):
        full_model.add(feat_extract_model.layers[i])

    full_model.add(Flatten())
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    full_model.compile(optimizer=sgd, loss='mse')

    return full_model


def batch_norm_vgg_model(shape, filter_list, maxpool=True, op_only_middle=True, batch_norm=True):
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    input_img = Input(shape=shape)

    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same', dim_ordering='tf')(input_img)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)

    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)

    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)

    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)

    if (op_only_middle):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)

    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)

    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)

    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)

    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)

    decoded = Convolution2D(shape[2], 3, 3, activation='relu', border_mode='same')(x1)

    autoencoder = Model(input_img, decoded)

    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)

    autoencoder.compile(optimizer=sgd, loss=loss_DSSIM_theano)

    return autoencoder


def GAN_model(shape, filter_list, maxpool=True, op_only_middle=True, batch_norm=False, autoencoder=False, highcap=False,
              dropout=True):
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    input_img = Input(shape=shape)

    x1 = GaussianNoise(0.05)(input_img)

    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same', dim_ordering='tf')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    if (op_only_middle):
        x1 = MaxPooling2D((2, 2), border_mode='same')(x1)
        x1 = GaussianNoise(0.05)(x1)

    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
    if (batch_norm):
        x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    if (highcap):
        x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
        if (batch_norm):
            x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
    if (maxpool):
        x1 = UpSampling2D((2, 2))(x1)
    if (dropout):
        x1 = Dropout(0.5)(x1)

    decoded = Convolution2D(shape[2], 3, 3, activation='tanh', border_mode='same')(x1)

    if (autoencoder):
        autoencoder = Model(input=input_img, output=decoded)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(input=[input_img, input_background], output=merged)

    input_img_y = Input(shape=shape)

    y1 = Convolution2D(filter_list[0], 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2))(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Convolution2D(filter_list[1], 3, 3, border_mode='same', subsample=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Convolution2D(filter_list[2], 3, 3, border_mode='same', subsample=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Convolution2D(filter_list[3], 3, 3, border_mode='same', subsample=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(input=input_img_y, output=y1)

    if (autoencoder):
        return autoencoder, gan_merged_model, adversary_model
    else:
        return gan_merged_model, adversary_model


def DCGAN_model(shape, filter_list, noise=True):
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x1 = GaussianNoise(0.05)(input_img)

    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same', dim_ordering='tf')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same', subsample=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same', subsample=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same', subsample=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same', subsample=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # THIS IS THE MIDDLE OF THE GENERATOR (OP SHAPE HERE IS [4,4,filter_list[3])
    x1 = GaussianNoise(0.05)(x1)

    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = UpSampling2D((2, 2))(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = UpSampling2D((2, 2))(x1)
    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Convolution2D(filter_list[0], 3, 3, border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = UpSampling2D((2, 2))(x1)
    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    decoded = Convolution2D(shape[2], 3, 3, border_mode='same', activation='tanh')(x1)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(input=[input_img, input_background], output=merged)

    input_img_y = Input(shape=shape)

    y1 = Convolution2D(filter_list[0], 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2))(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Convolution2D(filter_list[1], 3, 3, border_mode='same', subsample=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Convolution2D(filter_list[2], 3, 3, border_mode='same', subsample=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Convolution2D(filter_list[3], 3, 3, border_mode='same', subsample=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(input=input_img_y, output=y1)

    return gan_merged_model, adversary_model


def DCGAN_model_ker2(shape, filter_list, noise=True):
    # THIS IS IN KERAS 2.0.0 API
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x1 = GaussianNoise(0.05)(input_img)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # THIS IS THE MIDDLE OF THE GENERATOR (OP SHAPE HERE IS [4,4,filter_list[3])
    x1 = GaussianNoise(0.05)(x1)

    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2DTranspose(filters=filter_list[2], kernel_size=(3, 3), strides=2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2DTranspose(filters=filter_list[1], kernel_size=(3, 3), strides=2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2DTranspose(filters=filter_list[0], kernel_size=(3, 3), strides=2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    decoded = Conv2DTranspose(filters=shape[2], kernel_size=(3, 3), padding='same', activation='tanh')(x1)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(outputs=merged, inputs=[input_img, input_background])

    input_img_y = Input(shape=shape)

    y1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(outputs=y1, inputs=input_img_y)

    return gan_merged_model, adversary_model


def DCGAN_ker2_caption_LSTM(shape, filter_list, noise=True):
    # THIS IS IN KERAS 2.0.0 API
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x1 = GaussianNoise(0.05)(input_img)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    # THIS IS THE MIDDLE OF THE GENERATOR (OP SHAPE HERE IS [4,4,filter_list[3])
    x1 = GaussianNoise(0.05)(x1)

    input_caption_vector = Input(shape=(50, 300))
    lstm_layer = LSTM(64)(input_caption_vector)
    dense_layer = Dense(64)(lstm_layer)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense_layer)

    z1 = concatenate([x1, reshaped])
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[2], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[1], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[0], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    decoded = Conv2DTranspose(filters=shape[2], kernel_size=(3, 3), padding='same', activation='tanh')(z1)
    decoded_clarity = Conv2D(filters=shape[2], kernel_size=(1, 1), padding='same', activation='tanh')(decoded)

    input_background = Input(shape=shape)

    merged = merge([decoded_clarity, input_background],
                   mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(outputs=merged, inputs=[input_img, input_caption_vector, input_background])

    input_img_y = Input(shape=shape)

    y1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(outputs=y1, inputs=input_img_y)

    return gan_merged_model, adversary_model


def DCGAN_ker2_caption_LSTM_inception(shape, filter_list, noise=True):
    # THIS IS IN KERAS 2.0.0 API
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x1 = GaussianNoise(0.05)(input_img)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    # THIS IS THE MIDDLE OF THE GENERATOR (OP SHAPE HERE IS [4,4,filter_list[3])
    x1 = GaussianNoise(0.05)(x1)

    input_caption_vector = Input(shape=(50, 300))
    lstm_layer = LSTM(64)(input_caption_vector)
    dense_layer = Dense(64)(lstm_layer)
    dense_layer_with_noise = GaussianNoise(0.01)(dense_layer)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense_layer_with_noise)

    z1 = concatenate([x1, reshaped])
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[2], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[1], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[0], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    # Start adding inception Layers (z1->1*1 filters, 3*3 filters, 2*2 filters)

    # Inception 1
    inception1_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(z1)
    inception1_1x1 = BatchNormalization()(inception1_1x1)
    inception1_1x1_activate = LeakyReLU(0.2)(inception1_1x1)
    inception1_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(z1)
    inception1_2x2 = BatchNormalization()(inception1_2x2)
    inception1_2x2_activate = LeakyReLU(0.2)(inception1_2x2)
    inception1_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(z1)
    inception1_3x3 = BatchNormalization()(inception1_3x3)
    inception1_3x3_activate = LeakyReLU(0.2)(inception1_3x3)
    inception1_out = concatenate([inception1_1x1_activate, inception1_2x2_activate, inception1_3x3_activate])

    if (noise):
        inception1_out = GaussianNoise(0.01)(inception1_out)

    # Inception 2
    inception2_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(inception1_out)
    inception2_1x1 = BatchNormalization()(inception2_1x1)
    inception2_1x1_activate = LeakyReLU(0.2)(inception2_1x1)
    inception2_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(inception1_out)
    inception2_2x2 = BatchNormalization()(inception2_2x2)
    inception2_2x2_activate = LeakyReLU(0.2)(inception2_2x2)
    inception2_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(inception1_out)
    inception2_3x3 = BatchNormalization()(inception2_3x3)
    inception2_3x3_activate = LeakyReLU(0.2)(inception2_3x3)
    inception2_out = concatenate([inception2_1x1_activate, inception2_2x2_activate, inception2_3x3_activate])

    if (noise):
        inception2_out = GaussianNoise(0.01)(inception2_out)

    # Inception 3
    inception3_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(inception2_out)
    inception3_1x1 = BatchNormalization()(inception3_1x1)
    inception3_1x1_activate = LeakyReLU(0.2)(inception3_1x1)
    inception3_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(inception2_out)
    inception3_2x2 = BatchNormalization()(inception3_2x2)
    inception3_2x2_activate = LeakyReLU(0.2)(inception3_2x2)
    inception3_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(inception2_out)
    inception3_3x3 = BatchNormalization()(inception3_3x3)
    inception3_3x3_activate = LeakyReLU(0.2)(inception3_3x3)
    inception3_out = concatenate([inception3_1x1_activate, inception3_2x2_activate, inception3_3x3_activate])

    if (noise):
        inception3_out = GaussianNoise(0.01)(inception3_out)

    decoded = Conv2DTranspose(filters=shape[2], kernel_size=(3, 3), padding='same', activation='tanh')(inception3_out)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(outputs=merged, inputs=[input_img, input_caption_vector, input_background])

    input_img_y = Input(shape=shape)

    y1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(outputs=y1, inputs=input_img_y)

    return gan_merged_model, adversary_model


def DCGAN_ker2_caption_LSTM_inception_latent(shape, filter_list, noise=True):
    # THIS IS IN KERAS 2.0.0 API
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x1 = GaussianNoise(0.05)(input_img)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    # THIS IS THE MIDDLE OF THE GENERATOR (OP SHAPE HERE IS [4,4,filter_list[3])
    x1 = Flatten()(x1)
    x1 = GaussianNoise(0.05)(x1)
    x1 = Dense(512, activation='sigmoid')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(512, activation='sigmoid')(x1)
    x1 = BatchNormalization()(x1)

    input_caption_vector = Input(shape=(50, 300))
    lstm_layer = LSTM(128)(input_caption_vector)
    dense_layer = Dense(128, activation='sigmoid')(lstm_layer)
    dense_layer = BatchNormalization()(dense_layer)

    dense_layer_with_noise = GaussianNoise(0.01)(dense_layer)

    z1 = concatenate([x1, dense_layer_with_noise])  # Latent variables of shape (192,1)

    z1 = Reshape(target_shape=(4, 4, 40))(z1)

    # Start of reconstruction point.
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[3], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[2], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[1], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[0], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    # Start adding inception Layers (z1->1*1 filters, 3*3 filters, 2*2 filters)

    # Inception 1
    inception1_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(z1)
    inception1_1x1 = BatchNormalization()(inception1_1x1)
    inception1_1x1_activate = LeakyReLU(0.2)(inception1_1x1)
    inception1_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(z1)
    inception1_2x2 = BatchNormalization()(inception1_2x2)
    inception1_2x2_activate = LeakyReLU(0.2)(inception1_2x2)
    inception1_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(z1)
    inception1_3x3 = BatchNormalization()(inception1_3x3)
    inception1_3x3_activate = LeakyReLU(0.2)(inception1_3x3)
    inception1_out = concatenate([inception1_1x1_activate, inception1_2x2_activate, inception1_3x3_activate])

    if (noise):
        inception1_out = GaussianNoise(0.01)(inception1_out)

    # Inception 2
    inception2_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(inception1_out)
    inception2_1x1 = BatchNormalization()(inception2_1x1)
    inception2_1x1_activate = LeakyReLU(0.2)(inception2_1x1)
    inception2_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(inception1_out)
    inception2_2x2 = BatchNormalization()(inception2_2x2)
    inception2_2x2_activate = LeakyReLU(0.2)(inception2_2x2)
    inception2_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(inception1_out)
    inception2_3x3 = BatchNormalization()(inception2_3x3)
    inception2_3x3_activate = LeakyReLU(0.2)(inception2_3x3)
    inception2_out = concatenate([inception2_1x1_activate, inception2_2x2_activate, inception2_3x3_activate])

    if (noise):
        inception2_out = GaussianNoise(0.01)(inception2_out)

    # Inception 3
    inception3_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(inception2_out)
    inception3_1x1 = BatchNormalization()(inception3_1x1)
    inception3_1x1_activate = LeakyReLU(0.2)(inception3_1x1)
    inception3_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(inception2_out)
    inception3_2x2 = BatchNormalization()(inception3_2x2)
    inception3_2x2_activate = LeakyReLU(0.2)(inception3_2x2)
    inception3_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(inception2_out)
    inception3_3x3 = BatchNormalization()(inception3_3x3)
    inception3_3x3_activate = LeakyReLU(0.2)(inception3_3x3)
    inception3_out = concatenate([inception3_1x1_activate, inception3_2x2_activate, inception3_3x3_activate])

    if (noise):
        inception3_out = GaussianNoise(0.01)(inception3_out)

    decoded = Conv2DTranspose(filters=shape[2], kernel_size=(3, 3), padding='same', activation='tanh')(inception3_out)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(outputs=merged, inputs=[input_img, input_caption_vector, input_background])

    input_img_y = Input(shape=shape)

    y1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(outputs=y1, inputs=input_img_y)

    return gan_merged_model, adversary_model


def DC_caption_LSTM_inception_exact(shape, filter_list, noise=True):
    # THIS IS IN KERAS 2.0.0 API
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x01 = GaussianNoise(0.05)(input_img)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x01)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    # THIS IS THE MIDDLE OF THE GENERATOR (OP SHAPE HERE IS [4,4,filter_list[3])
    x1 = Flatten()(x1)
    x1 = GaussianNoise(0.05)(x1)
    x1 = Dense(512, activation='sigmoid')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(512, activation='sigmoid')(x1)
    x1 = BatchNormalization()(x1)

    input_caption_vector = Input(shape=(50, 300))
    lstm_layer = LSTM(128)(input_caption_vector)
    dense_layer = Dense(128, activation='sigmoid')(lstm_layer)
    dense_layer = BatchNormalization()(dense_layer)

    dense_layer_with_noise = GaussianNoise(0.01)(dense_layer)

    z1 = concatenate([x1, dense_layer_with_noise])  # Latent variables of shape (192,1)
    z1 = Dense(int(filter_list[3] * 4 * 4), activation='sigmoid')(z1)
    z1 = BatchNormalization()(z1)

    z1 = Reshape(target_shape=(4, 4, filter_list[3]))(z1)

    # Start of reconstruction point.
    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[2], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[2], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.02)(z1)

    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[1], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[1], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2D(filter_list[0], (3, 3), padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)
    z1 = Conv2DTranspose(filters=filter_list[0], kernel_size=(3, 3), strides=2, padding='same')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(0.2)(z1)

    if (noise):
        z1 = GaussianNoise(0.01)(z1)

    # Start adding inception Layers (z1->1*1 filters, 3*3 filters, 2*2 filters)

    # Inception 1
    inception1_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(z1)
    inception1_1x1 = BatchNormalization()(inception1_1x1)
    inception1_1x1_activate = LeakyReLU(0.2)(inception1_1x1)
    inception1_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(z1)
    inception1_2x2 = BatchNormalization()(inception1_2x2)
    inception1_2x2_activate = LeakyReLU(0.2)(inception1_2x2)
    inception1_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(z1)
    inception1_3x3 = BatchNormalization()(inception1_3x3)
    inception1_3x3_activate = LeakyReLU(0.2)(inception1_3x3)
    inception1_out = concatenate([inception1_1x1_activate, inception1_2x2_activate, inception1_3x3_activate])

    if (noise):
        inception1_out = GaussianNoise(0.01)(inception1_out)

    # Inception 2
    inception2_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(inception1_out)
    inception2_1x1 = BatchNormalization()(inception2_1x1)
    inception2_1x1_activate = LeakyReLU(0.2)(inception2_1x1)
    inception2_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(inception1_out)
    inception2_2x2 = BatchNormalization()(inception2_2x2)
    inception2_2x2_activate = LeakyReLU(0.2)(inception2_2x2)
    inception2_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(inception1_out)
    inception2_3x3 = BatchNormalization()(inception2_3x3)
    inception2_3x3_activate = LeakyReLU(0.2)(inception2_3x3)
    inception2_out = concatenate([inception2_1x1_activate, inception2_2x2_activate, inception2_3x3_activate])

    if (noise):
        inception2_out = GaussianNoise(0.01)(inception2_out)

    # Inception 3
    inception3_1x1 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same')(inception2_out)
    inception3_1x1 = BatchNormalization()(inception3_1x1)
    inception3_1x1_activate = LeakyReLU(0.2)(inception3_1x1)
    inception3_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same')(inception2_out)
    inception3_2x2 = BatchNormalization()(inception3_2x2)
    inception3_2x2_activate = LeakyReLU(0.2)(inception3_2x2)
    inception3_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same')(inception2_out)
    inception3_3x3 = BatchNormalization()(inception3_3x3)
    inception3_3x3_activate = LeakyReLU(0.2)(inception3_3x3)
    inception3_out = concatenate([inception3_1x1_activate, inception3_2x2_activate, inception3_3x3_activate])

    if (noise):
        inception3_out = GaussianNoise(0.01)(inception3_out)

    decoded = Conv2DTranspose(filters=shape[2], kernel_size=(3, 3), padding='same', activation='tanh')(inception3_out)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(outputs=merged, inputs=[input_img, input_caption_vector, input_background])

    input_img_y = Input(shape=shape)

    y1 = Conv2D(filter_list[0], (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(input_img_y)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[1], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[2], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(filter_list[3], (3, 3), padding='same', strides=(2, 2))(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model = Model(outputs=y1, inputs=input_img_y)

    return gan_merged_model, adversary_model


def DC_caption_LSTM_inception_only(shape, filter_list, train_both=False, noise=True):
    # THIS IS IN KERAS 2.0.0 API
    rows = shape[0]
    cols = shape[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # GENERATOR
    input_img = Input(shape=shape)

    x01 = GaussianNoise(0.05)(input_img)

    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x01)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Conv2D(filter_list[0], (3, 3), padding='same', data_format='channels_last')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(0.2)(x1)

    if (noise):
        x1 = GaussianNoise(0.02)(x1)

    # input size = (filter_list[0],64,64)
    # Inception convolution 1
    inceptionc1_1x1 = Conv2D(filters=int(filter_list[0]), kernel_size=(1, 1), padding='same',
                             data_format='channels_last')(x1)
    inceptionc1_1x1 = BatchNormalization()(inceptionc1_1x1)
    inceptionc1_1x1_activate = LeakyReLU(0.2)(inceptionc1_1x1)
    inceptionc1_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same',
                             data_format='channels_last')(x1)
    inceptionc1_3x3 = BatchNormalization()(inceptionc1_3x3)
    inceptionc1_3x3_activate = LeakyReLU(0.2)(inceptionc1_3x3)
    inceptionc1_5x5 = Conv2D(filters=int(filter_list[0] / 3), kernel_size=(5, 5), padding='same',
                             data_format='channels_last')(x1)
    inceptionc1_5x5 = BatchNormalization()(inceptionc1_5x5)
    inceptionc1_5x5_activate = LeakyReLU(0.2)(inceptionc1_5x5)
    inceptionc1_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x1)
    inceptionc1_pool = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(1, 1), padding='same',
                              data_format='channels_last')(inceptionc1_pool)
    inceptionc1_pool = BatchNormalization()(inceptionc1_pool)
    inceptionc1_pool_activate = LeakyReLU(0.2)(inceptionc1_pool)

    inceptionc1_out = concatenate(
        [inceptionc1_1x1_activate, inceptionc1_3x3_activate, inceptionc1_5x5_activate, inceptionc1_pool_activate])

    if (noise):
        inceptionc1_out = GaussianNoise(0.02)(inceptionc1_out)

    # Inception convolution 2
    inceptionc2_1x1 = Conv2D(filters=int(filter_list[1]), kernel_size=(1, 1), padding='same',
                             data_format='channels_last')(inceptionc1_out)
    inceptionc2_1x1 = BatchNormalization()(inceptionc2_1x1)
    inceptionc2_1x1_activate = LeakyReLU(0.2)(inceptionc2_1x1)
    inceptionc2_3x3 = Conv2D(filters=int(filter_list[1] / 2), kernel_size=(3, 3), padding='same',
                             data_format='channels_last')(inceptionc1_out)
    inceptionc2_3x3 = BatchNormalization()(inceptionc2_3x3)
    inceptionc2_3x3_activate = LeakyReLU(0.2)(inceptionc2_3x3)
    inceptionc2_5x5 = Conv2D(filters=int(filter_list[1] / 3), kernel_size=(5, 5), padding='same',
                             data_format='channels_last')(inceptionc1_out)
    inceptionc2_5x5 = BatchNormalization()(inceptionc2_5x5)
    inceptionc2_5x5_activate = LeakyReLU(0.2)(inceptionc2_5x5)
    inceptionc2_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(inceptionc1_out)
    inceptionc2_pool = Conv2D(filters=int(filter_list[1] / 2), kernel_size=(1, 1), padding='same',
                              data_format='channels_last')(inceptionc2_pool)
    inceptionc2_pool = BatchNormalization()(inceptionc2_pool)
    inceptionc2_pool_activate = LeakyReLU(0.2)(inceptionc2_pool)

    inceptionc2_out = concatenate(
        [inceptionc2_1x1_activate, inceptionc2_3x3_activate, inceptionc2_5x5_activate, inceptionc2_pool_activate])

    if (noise):
        inceptionc2_out = GaussianNoise(0.02)(inceptionc2_out)

    # Inception convolution 3
    inceptionc3_1x1 = Conv2D(filters=int(filter_list[2]), kernel_size=(1, 1), padding='same',
                             data_format='channels_last')(inceptionc2_out)
    inceptionc3_1x1 = BatchNormalization()(inceptionc3_1x1)
    inceptionc3_1x1_activate = LeakyReLU(0.2)(inceptionc3_1x1)
    inceptionc3_3x3 = Conv2D(filters=int(filter_list[2] / 2), kernel_size=(3, 3), padding='same',
                             data_format='channels_last')(inceptionc2_out)
    inceptionc3_3x3 = BatchNormalization()(inceptionc3_3x3)
    inceptionc3_3x3_activate = LeakyReLU(0.2)(inceptionc3_3x3)
    inceptionc3_5x5 = Conv2D(filters=int(filter_list[2] / 3), kernel_size=(5, 5), padding='same',
                             data_format='channels_last')(inceptionc2_out)
    inceptionc3_5x5 = BatchNormalization()(inceptionc3_5x5)
    inceptionc3_5x5_activate = LeakyReLU(0.2)(inceptionc3_5x5)
    inceptionc3_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(inceptionc2_out)
    inceptionc3_pool = Conv2D(filters=int(filter_list[2] / 2), kernel_size=(1, 1), padding='same',
                              data_format='channels_last')(inceptionc3_pool)
    inceptionc3_pool = BatchNormalization()(inceptionc3_pool)
    inceptionc3_pool_activate = LeakyReLU(0.2)(inceptionc3_pool)

    inceptionc3_out = concatenate(
        [inceptionc3_1x1_activate, inceptionc3_3x3_activate, inceptionc3_5x5_activate, inceptionc3_pool_activate])

    if (noise):
        inceptionc3_out = GaussianNoise(0.02)(inceptionc3_out)

    # Inception convolution 4
    inceptionc4_1x1 = Conv2D(filters=int(filter_list[3]), kernel_size=(1, 1), padding='same',
                             data_format='channels_last')(inceptionc3_out)
    inceptionc4_1x1 = BatchNormalization()(inceptionc4_1x1)
    inceptionc4_1x1_activate = LeakyReLU(0.2)(inceptionc4_1x1)
    inceptionc4_3x3 = Conv2D(filters=int(filter_list[3] / 2), kernel_size=(3, 3), padding='same',
                             data_format='channels_last')(inceptionc3_out)
    inceptionc4_3x3 = BatchNormalization()(inceptionc4_3x3)
    inceptionc4_3x3_activate = LeakyReLU(0.2)(inceptionc4_3x3)
    inceptionc4_5x5 = Conv2D(filters=int(filter_list[3] / 3), kernel_size=(5, 5), padding='same',
                             data_format='channels_last')(inceptionc3_out)
    inceptionc4_5x5 = BatchNormalization()(inceptionc4_5x5)
    inceptionc4_5x5_activate = LeakyReLU(0.2)(inceptionc4_5x5)
    inceptionc4_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(inceptionc3_out)
    inceptionc4_pool = Conv2D(filters=int(filter_list[3] / 2), kernel_size=(1, 1), padding='same',
                              data_format='channels_last')(inceptionc4_pool)
    inceptionc4_pool = BatchNormalization()(inceptionc4_pool)
    inceptionc4_pool_activate = LeakyReLU(0.2)(inceptionc4_pool)

    inceptionc4_out = concatenate(
        [inceptionc4_1x1_activate, inceptionc4_3x3_activate, inceptionc4_5x5_activate, inceptionc4_pool_activate])

    if (noise):
        inceptionc4_out = GaussianNoise(0.02)(inceptionc4_out)

    # SHAPE here(2*filter_list[3]+filter_list[3]/3,64,64)
    conv_out = Conv2D(filters=filter_list[3], kernel_size=(4, 4), padding='same', strides=(4, 4),
                      data_format='channels_last')(inceptionc4_out)
    conv_out = BatchNormalization()(conv_out)
    conv_out = LeakyReLU(0.2)(conv_out)

    # SHAPE_here(filter_list[3],16,16)
    conv_out = Conv2D(filters=(filter_list[3] - 16), kernel_size=(4, 4), padding='same', strides=(4, 4),
                      data_format='channels_last')(conv_out)
    conv_out = BatchNormalization()(conv_out)
    conv_out = LeakyReLU(0.2)(conv_out)
    # SHAPE_here(filter_list[3]-16,4,4)

    input_caption_vector = Input(shape=(50, 300))
    lstm_layer = LSTM(512)(input_caption_vector)
    dense_layer = Dense(256, activation='sigmoid')(lstm_layer)
    dense_layer = BatchNormalization()(dense_layer)

    captions_to_maps = Reshape(target_shape=(4, 4, 16))(dense_layer)

    conv_caption_merge_out = concatenate([conv_out, captions_to_maps])
    if (noise):
        conv_caption_merge_out = GaussianNoise(0.01)(conv_caption_merge_out)

    # SHAPE_here(filter_list[3],4,4)

    # Start of RECONSTRUCTION point.
    # Upsample First, Then use convolution.
    # Use only inception modules
    # Start adding inception Layers (z1->1*1 filters, 3*3 filters, 2*2 filters)

    # Inception 1
    upsampled_to_inception1 = UpSampling2D((2, 2))(conv_caption_merge_out)

    inception1_1x1 = Conv2D(filters=int(filter_list[2]), kernel_size=(1, 1), padding='same', strides=(1, 1),
                            data_format='channels_last')(upsampled_to_inception1)
    inception1_1x1 = BatchNormalization()(inception1_1x1)
    inception1_1x1_activate = LeakyReLU(0.2)(inception1_1x1)

    inception1_2x2 = Conv2D(filters=int(filter_list[2] / 2), kernel_size=(2, 2), padding='same',
                            data_format='channels_last')(upsampled_to_inception1)
    inception1_2x2 = BatchNormalization()(inception1_2x2)
    inception1_2x2_activate = LeakyReLU(0.2)(inception1_2x2)

    inception1_3x3 = Conv2D(filters=int(filter_list[2] / 2), kernel_size=(3, 3), padding='same',
                            data_format='channels_last')(upsampled_to_inception1)
    inception1_3x3 = BatchNormalization()(inception1_3x3)
    inception1_3x3_activate = LeakyReLU(0.2)(inception1_3x3)

    inception1_4x4 = Conv2D(filters=int(filter_list[2] / 2), kernel_size=(4, 4), padding='same',
                            data_format='channels_last')(upsampled_to_inception1)
    inception1_4x4 = BatchNormalization()(inception1_4x4)
    inception1_4x4_activate = LeakyReLU(0.2)(inception1_4x4)

    inception1_out = concatenate(
        [inception1_1x1_activate, inception1_2x2_activate, inception1_3x3_activate, inception1_4x4_activate])
    # inception1_out = Conv2D(filters=filter_list[2], kernel_size=(1, 1), padding='same', strides=(1, 1),
    #                         data_format='channels_last')(inception1_out)
    # inception1_out = BatchNormalization()(inception1_out)
    # inception1_out = LeakyReLU(0.2)(inception1_out)
    if (noise):
        inception1_out = GaussianNoise(0.01)(inception1_out)

    # SHAPE_here(filter_list[2],8,8)

    # Inception 2
    upsampled_to_inception2 = UpSampling2D((2, 2))(inception1_out)

    inception2_1x1 = Conv2D(filters=int(filter_list[1]), kernel_size=(1, 1), padding='same', strides=(1, 1),
                            data_format='channels_last')(upsampled_to_inception2)
    inception2_1x1 = BatchNormalization()(inception2_1x1)
    inception2_1x1_activate = LeakyReLU(0.2)(inception2_1x1)

    inception2_2x2 = Conv2D(filters=int(filter_list[1] / 2), kernel_size=(2, 2), padding='same',
                            data_format='channels_last')(upsampled_to_inception2)
    inception2_2x2 = BatchNormalization()(inception2_2x2)
    inception2_2x2_activate = LeakyReLU(0.2)(inception2_2x2)

    inception2_3x3 = Conv2D(filters=int(filter_list[1] / 2), kernel_size=(3, 3), padding='same',
                            data_format='channels_last')(upsampled_to_inception2)
    inception2_3x3 = BatchNormalization()(inception2_3x3)
    inception2_3x3_activate = LeakyReLU(0.2)(inception2_3x3)

    inception2_4x4 = Conv2D(filters=int(filter_list[1] / 2), kernel_size=(4, 4), padding='same',
                            data_format='channels_last')(upsampled_to_inception2)
    inception2_4x4 = BatchNormalization()(inception2_4x4)
    inception2_4x4_activate = LeakyReLU(0.2)(inception2_4x4)

    inception2_out = concatenate(
        [inception2_1x1_activate, inception2_2x2_activate, inception2_3x3_activate, inception2_4x4_activate])
    # inception2_out = Conv2D(filters=filter_list[1], kernel_size=(1, 1), padding='same', strides=(1, 1),
    #                         data_format='channels_last')(inception2_out)
    # inception2_out = BatchNormalization()(inception2_out)
    # inception2_out = LeakyReLU(0.2)(inception2_out)

    if (noise):
        inception2_out = GaussianNoise(0.01)(inception2_out)
    # SHAPE_here(filter_list[1],16,16)

    # Inception 3
    upsampled_to_inception3 = UpSampling2D((2, 2))(inception2_out)

    inception3_1x1 = Conv2D(filters=int(filter_list[0]), kernel_size=(1, 1), padding='same', strides=(1, 1),
                            data_format='channels_last')(upsampled_to_inception3)
    inception3_1x1 = BatchNormalization()(inception3_1x1)
    inception3_1x1_activate = LeakyReLU(0.2)(inception3_1x1)

    inception3_2x2 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(2, 2), padding='same',
                            data_format='channels_last')(upsampled_to_inception3)
    inception3_2x2 = BatchNormalization()(inception3_2x2)
    inception3_2x2_activate = LeakyReLU(0.2)(inception3_2x2)

    inception3_3x3 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(3, 3), padding='same',
                            data_format='channels_last')(upsampled_to_inception3)
    inception3_3x3 = BatchNormalization()(inception3_3x3)
    inception3_3x3_activate = LeakyReLU(0.2)(inception3_3x3)

    inception3_4x4 = Conv2D(filters=int(filter_list[0] / 2), kernel_size=(4, 4), padding='same',
                            data_format='channels_last')(upsampled_to_inception3)
    inception3_4x4 = BatchNormalization()(inception3_4x4)
    inception3_4x4_activate = LeakyReLU(0.2)(inception3_4x4)

    inception3_out = concatenate(
        [inception3_1x1_activate, inception3_2x2_activate, inception3_3x3_activate, inception3_4x4_activate])
    # inception3_out = Conv2D(filters=filter_list[0], kernel_size=(1, 1), padding='same', strides=(1, 1),
    #                         data_format='channels_last')(inception3_out)
    # inception3_out = BatchNormalization()(inception3_out)
    # inception3_out = LeakyReLU(0.2)(inception3_out)

    if (noise):
        inception3_out = GaussianNoise(0.01)(inception3_out)
    # SHAPE_here(filter_list[0],32,32)

    more_deconv = Conv2D(filters=int(filter_list[0]), kernel_size=(3, 3), padding='same',
                         data_format='channels_last')(inception3_out)
    more_deconv = BatchNormalization()(more_deconv)
    more_deconv = LeakyReLU(0.2)(more_deconv)
    more_deconv = Conv2D(filters=int(filter_list[0]), kernel_size=(3, 3), padding='same',
                         data_format='channels_last')(more_deconv)
    more_deconv = BatchNormalization()(more_deconv)
    more_deconv = LeakyReLU(0.2)(more_deconv)

    decoded = Conv2D(filters=shape[2], kernel_size=(3, 3), padding='same', activation='tanh',
                     data_format='channels_last')(more_deconv)
    # SHAPE_here(32,32,3)

    merged = merge([decoded, input_img], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(outputs=merged, inputs=[input_img, input_caption_vector])

    input_img_y = Input(shape=shape)

    y1 = Conv2D(int(filter_list[0] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(
        input_img_y)
    # y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(int(filter_list[1] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    # y1 = Conv2D(int(filter_list[1]*1.5), (3, 3), padding='same', strides=(1, 1), data_format='channels_last')(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(int(filter_list[2] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    # y1 = Conv2D(int(filter_list[2]*1.5), (3, 3), padding='same', strides=(1, 1), data_format='channels_last')(y1)
    # y1 = BatchNormalization()(y1)
    # y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Conv2D(int(filter_list[3] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(y1)
    y1 = BatchNormalization()(y1)
    y1 = LeakyReLU(alpha=0.2)(y1)

    y1 = Flatten()(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    adversary_model_full = Model(outputs=y1, inputs=input_img_y)

    z1 = Cropping2D(cropping=((16, 16), (16, 16)))(input_img_y)

    z1 = Conv2D(int(filter_list[0] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(z1)
    # y1 = BatchNormalization()(y1)
    z1 = LeakyReLU(alpha=0.2)(z1)

    z1 = Conv2D(int(filter_list[1] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(alpha=0.2)(z1)

    z1 = Conv2D(int(filter_list[2] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(alpha=0.2)(z1)

    z1 = Conv2D(int(filter_list[3] * 1.5), (3, 3), padding='same', strides=(2, 2), data_format='channels_last')(z1)
    z1 = BatchNormalization()(z1)
    z1 = LeakyReLU(alpha=0.2)(z1)

    z1 = Flatten()(z1)
    z1 = Dropout(0.5)(z1)
    z1 = Dense(1, activation='sigmoid')(z1)

    adversary_model_half = Model(outputs=z1, inputs=input_img_y)

    if (train_both):
        return gan_merged_model, adversary_model_full, adversary_model_half
    else:
        return gan_merged_model, adversary_model_full



objectives.loss_DSSIM_theano = loss_DSSIM_theano
