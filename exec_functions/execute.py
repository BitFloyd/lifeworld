import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.utils import shuffle

from data_package import data_fns
from model_pkg import model_zoo


def NOMAXPOOL():
    # Get data as numpy mem-map to not overload the RAM
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0

    # EXPERIMENT 1
    # --------------
    # VGG type model. Max pool in the middle of the representation to reduce
    # the size in half.


    print "GET MODEL"
    model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg_mp_False.model'

    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000001, verbose=1)

    # model = model_zoo.vgg_model(shape_img, filter_list=[64, 128, 256, 512], maxpool=False, op_only_middle=True,
    #                           highcap=False)

    if (os.path.isfile(model_filepath)):
        model = load_model(model_filepath)

    print "START FIT"
    history = model.fit(dset_middle_empty_train, dset_middle_train,
                        batch_size=100, nb_epoch=400,
                        callbacks=[es, checkpoint, reduce_lr],
                        validation_split=0.3,
                        shuffle=True)
    print "MAKE PREDICTIONS"
    predictions = model.predict(dset_middle_empty_val, batch_size=100)

    print "SAVE IMAGES"
    write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp01_maxpool_False_highcap_False_mp_middle/'
    data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path)


def SSIMLOSS():
    # Get data as numpy mem-map to not overload the RAM
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0

    # EXPERIMENT 1
    # --------------
    # VGG type model. Max pool in the middle of the representation to reduce
    # the size in half.


    print "GET MODEL"
    model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg_mp_T_SSIMLOSS.model'

    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000001, verbose=1)

    model_zoo.alpha = 1.0
    model_zoo.beta = 0.0
    model = model_zoo.vgg_model(shape_img, filter_list=[64, 128, 256, 512], maxpool=True, op_only_middle=True,
                                highcap=False)

    # if (os.path.isfile(model_filepath)):
    #     model = load_model(model_filepath)

    print "START FIT"
    history = model.fit(dset_middle_empty_train, dset_middle_train,
                        batch_size=100, nb_epoch=400,
                        callbacks=[es, checkpoint, reduce_lr],
                        validation_split=0.3,
                        shuffle=True)
    print "MAKE PREDICTIONS"
    predictions = model.predict(dset_middle_empty_val, batch_size=100)

    print "SAVE IMAGES"
    write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp01_mx_T_hcap_False_mp_middle_SSIMLOSS/'
    data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path)


def MSELOSS():
    # Get data as numpy mem-map to not overload the RAM
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0

    # EXPERIMENT 1
    # --------------
    # VGG type model. Max pool in the middle of the representation to reduce
    # the size in half.


    print "GET MODEL"
    model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg_mp_T_MSELOSS.model'

    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000001, verbose=1)

    model_zoo.alpha = 0.0
    model_zoo.beta = 1.0
    model = model_zoo.vgg_model(shape_img, filter_list=[64, 128, 256, 512], maxpool=True, op_only_middle=True,
                                highcap=False)

    # if (os.path.isfile(model_filepath)):
    #     model = load_model(model_filepath)

    print "START FIT"
    history = model.fit(dset_middle_empty_train, dset_middle_train,
                        batch_size=100, nb_epoch=400,
                        callbacks=[es, checkpoint, reduce_lr],
                        validation_split=0.3,
                        shuffle=True)
    print "MAKE PREDICTIONS"
    predictions = model.predict(dset_middle_empty_val, batch_size=100)

    print "SAVE IMAGES"
    write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp01_mx_T_hcap_False_mp_middle_MSELOSS/'
    data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path)


def FEATURE_MATCH():
    # Get data as numpy mem-map to not overload the RAM
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0

    # EXPERIMENT 3
    # --------------
    # Matching with features instead of pixels.
    # Idea : Features from reconstructed image should match the features from a pre-trained model


    print "GET MODEL"
    model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp03_vgg_mp_T_feature_match.model'

    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000001, verbose=1)

    full_model = model_zoo.feature_comparison_model(shape_img, [64, 128, 256, 512], maxpool=False, highcap=False,
                                                    op_only_middle=False)

    if (os.path.isfile(model_filepath)):
        full_model = load_model(model_filepath)

    feat_extract_model = VGG16(include_top=False, weights='imagenet', input_shape=shape_img)
    x = feat_extract_model.output
    x = model_zoo.Flatten()(x)
    feature_extraction_model_flat = model_zoo.Model(input=feat_extract_model.input, output=x)

    print "GET VGG FEATURES"
    train_feats = feature_extraction_model_flat.predict(dset_train, batch_size=200)

    print "START FIT"
    history = full_model.fit(dset_middle_empty_train, train_feats,
                             batch_size=200, nb_epoch=400,
                             callbacks=[es, checkpoint, reduce_lr],
                             validation_split=0.3,
                             shuffle=True)
    layer_index = 0
    for i in range(2, len(full_model.layers)):
        layer_shape = full_model.layers[i].output_shape
        if (layer_shape[1:] == shape_img):
            layer_index = i
            break

    test_model = model_zoo.Model(input=full_model.input,
                                 output=full_model.layers[layer_index].get_output_at(0))

    print "MAKE PREDICTIONS"
    predictions = test_model.predict(dset_middle_empty_val, batch_size=100)

    print "SAVE IMAGES"
    write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp03_vgg_mp_T_feature_match/'
    data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)


def GAN(num_runs=10, n_epoch=400, small_dset=False, load_saved_model=False, dset_length=10, batch_norm=False,
        auto=False):
    # Get data as numpy mem-map to not overload the RAM
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_train[i][start:end, start:end, :] = 0.0

    # EXPERIMENT 4
    # --------------
    # GAN 01

    #################################
    # Make datasets smaller for test#
    #################################
    if (small_dset):
        dset_train = dset_train[0:dset_length]
        dset_middle_train = dset_middle_train[0:dset_length]
        dset_middle_empty_train = dset_middle_empty_train[0:dset_length]
        dset_val = dset_val[0:dset_length]
        dset_middle_val = dset_middle_val[0:dset_length]
        dset_middle_empty_val = dset_middle_empty_val[0:dset_length]

    print "GET MODEL"
    model_filepath_adversary = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp04_gan_adversary.model'
    model_filepath_full = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp04_gan_full.model'

    es_adversary = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)

    checkpoint0 = ModelCheckpoint(model_filepath_adversary, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')

    checkpoint1 = ModelCheckpoint(model_filepath_full, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001, verbose=1)

    es_full_model = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)

    if (auto):
        autoencoder, gan_merged_model, adversary_model = model_zoo.GAN_model(shape_img, [64, 128, 256, 512],
                                                                             maxpool=True,
                                                                             op_only_middle=True,
                                                                             batch_norm=batch_norm,
                                                                             autoencoder=True)
    else:
        gan_merged_model, adversary_model = model_zoo.GAN_model(shape_img, [64, 128, 256, 512],
                                                                maxpool=True,
                                                                op_only_middle=True,
                                                                batch_norm=batch_norm)

    if (load_saved_model):
        if (os.path.isfile(model_filepath_adversary)):
            adversary_model = load_model(model_filepath_adversary)
        if (os.path.isfile(model_filepath_full)):
            full_model = load_model(model_filepath_full)

    sgd = model_zoo.SGD(lr=0.001, momentum=0.9, nesterov=True)

    if (auto):
        model_zoo.alpha = 0.5
        model_zoo.beta = 0.5

        print "COMPILE AUTOENCODER MODEL"
        autoencoder.compile(optimizer=sgd, loss=model_zoo.loss_DSSIM_theano)
        autoencoder.summary()
        print "####################################"

    print "COMPILE GAN MERGED MODEL"
    gan_merged_model.compile(optimizer=sgd, loss=model_zoo.loss_DSSIM_theano)
    gan_merged_model.summary()
    print "####################################"

    print "COMPILE ADVERSARY MODEL"
    adversary_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model.summary()
    print "####################################"

    print "COMPILE FULL MODEL"
    full_model_tensor = adversary_model(gan_merged_model.output)
    full_model = model_zoo.Model(gan_merged_model.input, full_model_tensor)
    full_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    full_model.summary()
    print "####################################"

    if (auto):
        print "START TRAINING AUTOENCODER"

        autoencoder.fit(dset_middle_empty_train, dset_middle_train,
                        batch_size=100,
                        nb_epoch=n_epoch,
                        callbacks=[es_full_model, reduce_lr],
                        validation_split=0.3,
                        shuffle=True,
                        verbose=True)
        print "FINISH AUTOENCODER TRAINING"

    print "START TRAINING RUNS"

    for i in range(0, num_runs):
        es_adversary.best = np.inf
        checkpoint0.best = np.inf
        es_full_model.best = np.inf
        checkpoint1.best = np.inf

        print "Run number: ", i + 1
        predictions = gan_merged_model.predict([dset_middle_empty_train, dset_middle_empty_train], batch_size=100,
                                               verbose=True)
        dset_max = np.vstack((dset_train, predictions))
        dset_real_false = np.zeros(((predictions.shape[0] + dset_train.shape[0]), 1))
        dset_real_false[0:dset_train.shape[0]] = 1
        dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

        print "Start Adversary Training"
        model_zoo.make_trainable(adversary_model, True)

        adversary_model.summary()
        print "#########################"
        print "Start Fit"
        print "#########################"
        adversary_model.fit(dset_max, dset_real_false,
                            batch_size=100,
                            nb_epoch=n_epoch,
                            callbacks=[es_adversary, checkpoint0, reduce_lr],
                            validation_split=0.1,
                            shuffle=True,
                            verbose=True)

        model_zoo.make_trainable(adversary_model, False)

        full_model.summary()
        print "#########################"
        print "Start Full Model Training"
        print "#########################"
        full_model.fit([dset_middle_empty_train, dset_middle_empty_train],
                       np.ones((dset_middle_empty_train.shape[0], 1)),
                       batch_size=100,
                       nb_epoch=n_epoch,
                       callbacks=[es_full_model, checkpoint1, reduce_lr],
                       validation_split=0.3,
                       shuffle=True,
                       verbose=True)

        gan_merged_model.summary()
        print "#########################"
        print "MAKE PREDICTIONS"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_val, dset_middle_empty_val], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_' + str(i + 1).zfill(
            2) + '/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

    print "FINISHED ALL RUNS"
