import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, merge, Dropout
from keras.layers import Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD
from theano.tensor import set_subtensor
from theano.tensor.nnet.neighbours import images2neibs
from keras import objectives
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


def GAN_model(shape, filter_list, maxpool=True, op_only_middle=True, batch_norm=False, autoencoder=False):
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

    if (autoencoder):
        autoencoder = Model(input=input_img, output=decoded)

    input_background = Input(shape=shape)

    merged = merge([decoded, input_background], mode=lambda x: set_subtensor(x[1][:, start:end, start:end, :], x[0]),
                   output_shape=lambda x: x[1])

    gan_merged_model = Model(input=[input_img, input_background], output=merged)

    input_img_y = Input(shape=shape)

    y1 = Convolution2D(filter_list[0], 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img_y)
    y1 = BatchNormalization()(y1)

    if (maxpool):
        y1 = MaxPooling2D((2, 2), border_mode='same')(y1)

    y1 = Convolution2D(filter_list[1], 3, 3, border_mode='same')(y1)
    if (batch_norm):
        y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    if (maxpool):
        y1 = MaxPooling2D((2, 2), border_mode='same')(y1)

    y1 = Convolution2D(filter_list[2], 3, 3, border_mode='same')(y1)
    if (batch_norm):
        y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    if (maxpool):
        y1 = MaxPooling2D((2, 2), border_mode='same')(y1)

    y1 = Convolution2D(filter_list[3], 3, 3, border_mode='same')(y1)
    if (batch_norm):
        y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    if (maxpool):
        y1 = MaxPooling2D((2, 2), border_mode='same')(y1)

    y1 = Flatten()(y1)
    y1 = Dense(2048, activation='sigmoid')(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(2048, activation='sigmoid')(y1)
    y1 = Dense(1, activation='sigmoid')(y1)

    # vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=shape)
    # x = Flatten()(vgg19.output)
    # x = Dense(2048, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(2048, activation='sigmoid')(x)
    # x = Dense(1, activation='sigmoid')(x)
    # adversary_model = Model(input = vgg19.input, output = x)

    adversary_model = Model(input=input_img_y, output=y1)

    if (autoencoder):
        return autoencoder, gan_merged_model, adversary_model
    else:
        return gan_merged_model, adversary_model


objectives.loss_DSSIM_theano = loss_DSSIM_theano
