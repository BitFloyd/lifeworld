from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
import keras.backend as K
from keras.optimizers import SGD
from theano.tensor.nnet.neighbours import images2neibs

alpha = 0.5
beta = 1 - alpha


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
