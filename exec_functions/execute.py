import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.utils import shuffle
from tqdm import tqdm

from data_package import data_fns
from model_pkg import model_zoo


def plot_loss(losses, per_frq=False):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminative loss')
    plt.plot(losses["g"], label='generative loss')
    plt.plot(losses["m"], label='ms/ssim loss')
    plt.legend()
    if (per_frq):
        plt.savefig('/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/dcg_losses_pf.png')
    else:
        plt.savefig('/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/dcg_losses.png')
    plt.close(fig)


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
        dset_middle_empty_val[i][start:end, start:end, :] = 0.0

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

    es_auto = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=50, verbose=1)

    es_adversary = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25, verbose=1)

    checkpoint0 = ModelCheckpoint(model_filepath_adversary, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')

    checkpoint1 = ModelCheckpoint(model_filepath_full, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=0.00000001, verbose=1)

    es_full_model = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25, verbose=1)

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
        model_zoo.alpha = 0.8
        model_zoo.beta = 0.2

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
                        callbacks=[es_auto],
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
                            callbacks=[es_adversary, checkpoint0],
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
                       callbacks=[es_full_model, checkpoint1],
                       validation_split=0.3,
                       shuffle=True,
                       verbose=True)

        gan_merged_model.summary()
        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_train, dset_middle_empty_train], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_train' + str(
            i + 1).zfill(
            2) + '/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_val, dset_middle_empty_val], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_test' + str(
            i + 1).zfill(
            2) + '/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

    print "FINISHED ALL RUNS"


def GAN_start_random(num_runs=10, n_epoch=400, small_dset=False, load_saved_model=False, dset_length=10,
                     batch_norm=False):
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
        dset_middle_empty_val[i][start:end, start:end, :] = 0.0

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

    es_adversary = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25, verbose=1)

    checkpoint0 = ModelCheckpoint(model_filepath_adversary, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')

    checkpoint1 = ModelCheckpoint(model_filepath_full, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')

    es_full_model = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25, verbose=1)

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

    print "START TRAINING RUNS"
    for i in range(0, num_runs):
        es_adversary.best = np.inf
        checkpoint0.best = np.inf
        es_full_model.best = np.inf
        checkpoint1.best = np.inf

        print "Run number: ", i + 1
        if (i > 0):
            predictions = gan_merged_model.predict([dset_middle_empty_train, dset_middle_empty_train], batch_size=100,
                                                   verbose=True)
        else:
            predictions = np.copy(dset_middle_empty_train)
            for j in range(0, len(predictions)):
                predictions[j][start:end, start:end, :] = np.random.rand(end - start, end - start,
                                                                         shape_img[2]) * 2.0 - 1.0

        dset_max = np.vstack((dset_train, predictions))
        dset_real_false = np.zeros(((predictions.shape[0] + dset_train.shape[0]), 1))
        dset_real_false[0:dset_train.shape[0]] = 1
        dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

        model_zoo.shuffle_weights(gan_merged_model)

        print "Start Adversary Training"
        model_zoo.make_trainable(adversary_model, True)

        adversary_model.summary()
        model_zoo.shuffle_weights(adversary_model)
        print "#########################"
        print "Start Fit"
        print "#########################"
        adversary_model.fit(dset_max, dset_real_false,
                            batch_size=100,
                            nb_epoch=n_epoch,
                            callbacks=[es_adversary, checkpoint0],
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
                       callbacks=[es_full_model, checkpoint1],
                       validation_split=0.3,
                       shuffle=True,
                       verbose=True)

        gan_merged_model.summary()
        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_train, dset_middle_empty_train], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_train' + str(
            i + 1).zfill(
            2) + '/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_val, dset_middle_empty_val], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_test' + str(
            i + 1).zfill(
            2) + '/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

    print "FINISHED ALL RUNS"


def train_for_n(dset_train, dset_middle_empty_train, adversary_model, gan_merged_model, full_model, losses,
                losses_per_frq,
                nb_epoch=50000, plt_frq=100, batch_size=50, predict_frq=1000, run_num=10, m_train=False):
    for e in tqdm(range(nb_epoch)):

        corrupt = int(0.1 * batch_size)
        rand_index = np.random.randint(0, dset_train.shape[0], size=batch_size)
        train_batch = dset_train[rand_index]
        middle_empty_train_batch = dset_middle_empty_train[rand_index]

        predictions = gan_merged_model.predict([middle_empty_train_batch, middle_empty_train_batch],
                                               batch_size=batch_size, verbose=False)

        dset_max = np.vstack((train_batch, predictions))

        # make soft labels with label smoothing
        # False = 0.0-0.3
        # True = 0.7-1.2

        dset_real_false = np.random.rand((predictions.shape[0] + train_batch.shape[0]), 1) * 0.3

        dset_real_false[0:train_batch.shape[0]] = np.random.rand(train_batch.shape[0], 1) * 0.5 + 0.7

        # corrupt labels with low probability for discriminator training.
        # i.e, 10% of batch size, make true labels as false and false labels as true.
        rand_index_label_corrupt = np.random.randint(0, train_batch.shape[0], corrupt)
        dset_real_false[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.3
        rand_index_label_corrupt = np.random.randint(train_batch.shape[0], dset_real_false.shape[0], corrupt)
        dset_real_false[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.5 + 0.7

        dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

        # discriminator training
        model_zoo.make_trainable(adversary_model, True)
        adversary_loss = adversary_model.train_on_batch(dset_max, dset_real_false)

        losses["d"].append(adversary_loss[0])
        losses_per_frq["d"].append(adversary_loss[0])

        model_zoo.make_trainable(adversary_model, False)

        # corrupt labels for generator training also, low_prob.
        generator_labels = np.random.rand(middle_empty_train_batch.shape[0], 1) * 0.5 + 0.7
        rand_index_label_corrupt = np.random.randint(0, middle_empty_train_batch.shape[0], corrupt)
        generator_labels[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.3

        # generator training
        g_loss = full_model.train_on_batch([middle_empty_train_batch, middle_empty_train_batch],
                                           generator_labels)
        losses["g"].append(g_loss[0])
        losses_per_frq["g"].append(g_loss[0])

        if (m_train):
            m_loss = gan_merged_model.train_on_batch([middle_empty_train_batch, middle_empty_train_batch], train_batch)
            losses["m"].append(m_loss)
            losses_per_frq["m"].append(m_loss)
        else:
            losses["m"].append(0.0)
            losses_per_frq["m"].append(0.0)

        if e % plt_frq == plt_frq - 1:
            plot_loss(losses, per_frq=False)
            plot_loss(losses_per_frq, per_frq=True)
            losses_per_frq = {"d": [], "g": [], "m": []}

        if e % predict_frq == predict_frq - 1:
            predictions = gan_merged_model.predict([middle_empty_train_batch, middle_empty_train_batch], batch_size=100)
            write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp05_dcgan_run_train_batch' + \
                         str(run_num + 1).zfill(2) + '_' + str(e) + '/'

            if not os.path.exists(write_path):
                os.makedirs(write_path)
            data_fns.write_predicted(train_batch, middle_empty_train_batch, predictions, write_path, middle_only=False)


def GAN_slow_converge(num_runs=10, n_epoch=1, batch_norm=False):
    # Get data as numpy mem-map to not overload the RAM
    plt.ioff()
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
        dset_middle_empty_val[i][start:end, start:end, :] = 0.0

    # EXPERIMENT 4
    # --------------
    # GAN 01

    print "GET MODEL"

    gan_merged_model, adversary_model = model_zoo.GAN_model(shape_img, [128, 256, 512, 1024],
                                                            maxpool=True,
                                                            op_only_middle=True,
                                                            highcap=True,
                                                            dropout=False,
                                                            batch_norm=batch_norm)

    losses = {"d": [], "g": []}
    losses_per_frq = {"d": [], "g": []}

    # sgd = model_zoo.SGD(lr=0.001, momentum=0.9, nesterov=True)
    adam = model_zoo.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9, epsilon=1e-8, decay=0.0)

    print "COMPILE ADVERSARY MODEL"
    adversary_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model.summary()
    print "####################################"

    print "COMPILE FULL MODEL"
    full_model_tensor = adversary_model(gan_merged_model.output)
    full_model = model_zoo.Model(gan_merged_model.input, full_model_tensor)
    full_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    full_model.summary()
    print "####################################"

    predictions = np.copy(dset_middle_empty_train)
    for j in range(0, len(predictions)):
        predictions[j][start:end, start:end, :] = np.random.rand(end - start, end - start, shape_img[2]) * 2.0 - 1.0

    dset_max = np.vstack((dset_train, predictions))
    dset_real_false = np.zeros(((predictions.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model, True)
    adversary_model.summary()

    print "#########################"
    print "Start ADVERSARY Fit INITIAL"
    print "#########################"

    adversary_model.fit(dset_max, dset_real_false,
                        batch_size=100,
                        nb_epoch=n_epoch,
                        validation_split=0.1,
                        shuffle=True,
                        verbose=True)

    print "START SLOW CONVERGE BATCH TRAINING"

    for i in range(0, num_runs):

        print "###########################"
        print i + 1
        print "###########################"

        train_for_n(dset_train,
                    dset_middle_empty_train,
                    adversary_model,
                    gan_merged_model,
                    full_model,
                    losses,
                    losses_per_frq,
                    nb_epoch=6000,
                    plt_frq=100,
                    batch_size=64,
                    predict_frq=1000,
                    run_num=i,
                    w_shuffle=False)

        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_train, dset_middle_empty_train], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_train' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"

        predictions = gan_merged_model.predict([dset_middle_empty_val, dset_middle_empty_val], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp04_gan_run_test' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

    print "FINISHED ALL RUNS"


def DCGAN_slow_converge(num_runs=10, n_epoch=1):
    # Get data as numpy mem-map to not overload the RAM
    plt.ioff()
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')

    del (dset_middle_val)
    del (dset_middle_train)

    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # EXPERIMENT 5
    # --------------
    # DC-GAN WITH ALL THE GAN-HACKS

    dset_middle_train_p = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/dset_middle_train_p.npy')
    dset_middle_val_p = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/dset_middle_val_p.npy')

    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = dset_middle_train_p[i]
        # dset_middle_empty_train[i][start:end, start:end, :] = np.random.rand(32,32,3)*2.0 - 1
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_val[i][start:end, start:end, :] = dset_middle_val_p[i]
        # dset_middle_empty_val[i][start:end, start:end, :] = np.random.rand(32, 32, 3) * 2.0 - 1

    print "DELETE SCRAP DATASETS"
    del (dset_middle_train_p)
    del (dset_middle_val_p)

    print "GET GAN MODEL"

    gan_merged_model, adversary_model = model_zoo.DCGAN_model_ker2(shape_img, [128, 256, 512, 1024], noise=True)

    losses = {"d": [], "g": [], "m": []}
    losses_per_frq = {"d": [], "g": [], "m": []}

    sgd = model_zoo.SGD(lr=0.0002, momentum=0.5, nesterov=True)
    adam = model_zoo.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9, epsilon=1e-8, decay=0.0)

    print "COMPILE GAN MERGED MODEL"
    # The loss is reduced so as to not completely constrain the filters to overfit to the training data
    model_zoo.alpha = 0.3 / 4  # SSIM and MSE loss
    model_zoo.beta = 0.7 / 4  # MSE loss kept higher to avoid patchy reconstructions

    gan_merged_model.compile(optimizer=adam, loss=model_zoo.loss_DSSIM_theano)
    gan_merged_model.summary()
    print "####################################"

    print "COMPILE ADVERSARY MODEL"
    adversary_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model.summary()
    print "####################################"

    print "COMPILE FULL MODEL"
    full_model_tensor = adversary_model(gan_merged_model.output)
    full_model = model_zoo.Model(outputs=full_model_tensor, inputs=gan_merged_model.input)
    full_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    full_model.summary()
    print "####################################"

    dset_max = np.vstack((dset_train, dset_middle_empty_train))
    dset_real_false = np.zeros(((dset_middle_empty_train.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model, True)
    adversary_model.summary()

    print "#########################"
    print "Start ADVERSARY Fit INITIAL"
    print "#########################"

    adversary_model.fit(dset_max, dset_real_false,
                        batch_size=100,
                        nb_epoch=n_epoch,
                        shuffle=True,
                        verbose=True)

    del (dset_max)
    del (dset_real_false)

    print "START SLOW CONVERGE BATCH TRAINING"

    for i in range(0, num_runs):

        print "###########################"
        print i + 1
        print "###########################"

        train_for_n(dset_train,
                    dset_middle_empty_train,
                    adversary_model,
                    gan_merged_model,
                    full_model,
                    losses,
                    losses_per_frq,
                    nb_epoch=6000,
                    plt_frq=100,
                    batch_size=64,
                    predict_frq=1000,
                    run_num=i, m_train=True)

        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        predictions = gan_merged_model.predict([dset_middle_empty_train, dset_middle_empty_train], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp05_dcgan_run_train' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"

        predictions = gan_merged_model.predict([dset_middle_empty_val, dset_middle_empty_val], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp05_dcgan_run_test' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

    print "FINISHED ALL RUNS"


def train_for_n_captions(dset_train, dset_middle_empty_train, train_captions, adversary_model, gan_merged_model,
                         full_model, losses,
                         losses_per_frq,
                         nb_epoch=50000, plt_frq=100, batch_size=50, predict_frq=1000, run_num=10, m_train=False,
                         cp=0.1):
    rand_disc_train = np.random.randint(0, nb_epoch, size=int(0.01 * nb_epoch))
    for e in tqdm(range(nb_epoch)):
        corrupt = int(cp * batch_size)
        rand_index = np.random.randint(0, dset_train.shape[0], size=batch_size)
        train_batch = dset_train[rand_index]
        middle_empty_train_batch = dset_middle_empty_train[rand_index]
        rand_caption_idx = np.random.randint(0, train_captions.shape[1], len(rand_index))
        captions_train_batch = train_captions[rand_index, rand_caption_idx]

        predictions = gan_merged_model.predict(
            [middle_empty_train_batch, captions_train_batch, middle_empty_train_batch],
            batch_size=batch_size, verbose=False)

        dset_max = np.vstack((train_batch, predictions))

        # make soft labels with label smoothing
        # False = 0.0-0.3
        # True = 0.7-1.2

        dset_real_false = np.random.rand((predictions.shape[0] + train_batch.shape[0]), 1) * 0.3

        dset_real_false[0:train_batch.shape[0]] = np.random.rand(train_batch.shape[0], 1) * 0.5 + 0.7

        # corrupt labels with low probability for discriminator training.
        # i.e, 10% of batch size, make true labels as false and false labels as true.
        rand_index_label_corrupt = np.random.randint(0, train_batch.shape[0], corrupt)
        dset_real_false[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.3
        rand_index_label_corrupt = np.random.randint(train_batch.shape[0], dset_real_false.shape[0], corrupt)
        dset_real_false[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.5 + 0.7

        dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

        model_zoo.make_trainable(adversary_model, True)
        adversary_loss = adversary_model.train_on_batch(dset_max, dset_real_false)

        losses["d"].append(adversary_loss[0])
        losses_per_frq["d"].append(adversary_loss[0])

        model_zoo.make_trainable(adversary_model, False)

        if e in rand_disc_train:
            # randomly train discriminator more twice for stability
            losses["g"].append(losses["g"][-1])
            losses["m"].append(losses["m"][-1])
            continue

        generator_labels = np.random.rand(middle_empty_train_batch.shape[0], 1) * 0.5 + 0.7
        rand_index_label_corrupt = np.random.randint(0, middle_empty_train_batch.shape[0], corrupt)
        generator_labels[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.3

        g_loss = full_model.train_on_batch([middle_empty_train_batch, captions_train_batch, middle_empty_train_batch],
                                           generator_labels)
        losses["g"].append(g_loss[0])
        losses_per_frq["g"].append(g_loss[0])

        if (m_train):
            m_loss = gan_merged_model.train_on_batch(
                [middle_empty_train_batch, captions_train_batch, middle_empty_train_batch], train_batch)
            losses["m"].append(m_loss)
            losses_per_frq["m"].append(m_loss)
        else:
            losses["m"].append(0.0)
            losses_per_frq["m"].append(0.0)

        if e % plt_frq == plt_frq - 1:
            plot_loss(losses, per_frq=False)
            plot_loss(losses_per_frq, per_frq=True)
            losses_per_frq = {"d": [], "g": [], "m": []}

        if e % predict_frq == predict_frq - 1:
            predictions = gan_merged_model.predict(
                [middle_empty_train_batch, captions_train_batch, middle_empty_train_batch], batch_size=100)
            write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp07_dcgan_caption_run_train_batch' + \
                         str(run_num + 1).zfill(2) + '_' + str(e) + '/'

            if not os.path.exists(write_path):
                os.makedirs(write_path)
            data_fns.write_predicted(train_batch, middle_empty_train_batch, predictions, write_path, middle_only=False)


def DCGAN_avec_captions_slow_converge(num_runs=10, n_epoch=1):
    # Get data as numpy mem-map to not overload the RAM
    plt.ioff()
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')

    del (dset_middle_val)
    del (dset_middle_train)

    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # EXPERIMENT 6
    # --------------
    # DC-GAN WITH ALL THE GAN-HACKS
    # Deconvolutions instead of upsampling
    # Takes captions as well.
    # For each batch training, any 1 out of the 5 captions are passed
    # For prediction, any of the captions may be used.

    dset_middle_train_p = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/dset_middle_train_p.npy')
    dset_middle_val_p = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/dset_middle_val_p.npy')

    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = dset_middle_train_p[i] * 2.0 - 1
        # dset_middle_empty_train[i][start:end, start:end, :] = np.random.rand(32,32,3)*2.0 - 1
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_val[i][start:end, start:end, :] = dset_middle_val_p[i] * 2.0 - 1
        # dset_middle_empty_val[i][start:end, start:end, :] = np.random.rand(32, 32, 3) * 2.0 - 1

    print "DELETE SCRAP DATASETS"
    del (dset_middle_train_p)
    del (dset_middle_val_p)

    print "##################"
    print "GET CAPTIONS"
    print "##################"
    train_captions = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/all_words_2_vectors_train.npy',
                             mmap_mode='r+')
    val_captions = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/all_words_2_vectors_val.npy',
                           mmap_mode='r+')

    print "GET GAN MODEL"
    gan_merged_model, adversary_model = model_zoo.DCGAN_ker2_caption_LSTM(shape_img, [128, 256, 512, 1024], noise=True)

    losses = {"d": [], "g": [], "m": []}
    losses_per_frq = {"d": [], "g": [], "m": []}

    sgd = model_zoo.SGD(lr=0.0002, momentum=0.5, nesterov=True)
    adam = model_zoo.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9, epsilon=1e-8, decay=0.0)

    print "COMPILE GAN MERGED MODEL"
    # The loss is reduced so as to not completely constrain the filters to overfit to the training data
    model_zoo.alpha = 0.3 / 4  # SSIM and MSE loss
    model_zoo.beta = 0.7 / 4  # MSE loss kept higher to avoid patchy reconstructions

    gan_merged_model.compile(optimizer=adam, loss=model_zoo.loss_DSSIM_theano)
    gan_merged_model.summary()
    print "####################################"

    print "COMPILE ADVERSARY MODEL"
    adversary_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model.summary()
    print "####################################"

    print "COMPILE FULL MODEL"
    full_model_tensor = adversary_model(gan_merged_model.output)
    full_model = model_zoo.Model(outputs=full_model_tensor, inputs=gan_merged_model.input)
    full_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    full_model.summary()
    model_weights = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/full_model_weights.h5'
    print "####################################"

    dset_max = np.vstack((dset_train, dset_middle_empty_train))
    dset_real_false = np.zeros(((dset_middle_empty_train.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model, True)
    adversary_model.summary()

    print "#########################"
    print "Start ADVERSARY Fit INITIAL"
    print "#########################"

    adversary_model.fit(dset_max, dset_real_false,
                        batch_size=100,
                        nb_epoch=n_epoch,
                        shuffle=True,
                        verbose=True)

    del (dset_max)
    del (dset_real_false)

    print "START SLOW CONVERGE BATCH TRAINING"

    for i in range(0, num_runs):

        print "###########################"
        print i + 1
        print "###########################"
        cp = 0.15 - 0.01 * (i)

        if (cp < 0.0):
            cp = 0.0

        train_for_n_captions(dset_train,
                             dset_middle_empty_train,
                             train_captions,
                             adversary_model,
                             gan_merged_model,
                             full_model,
                             losses,
                             losses_per_frq,
                             nb_epoch=6000,
                             plt_frq=100,
                             batch_size=64,
                             predict_frq=1000,
                             run_num=i, m_train=True,
                             cp=cp)

        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        captions = np.zeros((train_captions.shape[0], train_captions.shape[2], train_captions.shape[3]))
        for j in range(0, len(train_captions)):
            captions[j] = train_captions[j, np.random.randint(0, train_captions.shape[1], 1)]

        predictions = gan_merged_model.predict([dset_middle_empty_train, captions, dset_middle_empty_train],
                                               batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp06_dcgan_captions_run_train' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"
        captions = np.zeros((val_captions.shape[0], val_captions.shape[2], val_captions.shape[3]))
        for j in range(0, len(val_captions)):
            captions[j] = val_captions[j, np.random.randint(0, val_captions.shape[1], 1)]
        predictions = gan_merged_model.predict([dset_middle_empty_val, captions, dset_middle_empty_val], batch_size=100)

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp06_dcgan_captions_run_test' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

        del (captions)
        del (predictions)
        full_model.save_weights(model_weights)
        full_model.save('/usr/local/data/sejacob/lifeworld/data/inpainting/models/caption_full_model.h5')

    print "FINISHED ALL RUNS"


def DCGAN_avec_captions_slow_converge_avec_inception(num_runs=10, n_epoch=1):
    # Get data as numpy mem-map to not overload the RAM
    plt.ioff()
    print "GET_DATASETS"
    save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')

    del (dset_middle_val)
    del (dset_middle_train)

    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # EXPERIMENT 7
    # --------------
    # DC-GAN WITH ALL THE GAN-HACKS and inception modules for inpainting
    # Deconvolutions instead of upsampling
    # Takes captions as well.
    # For each batch training, any 1 out of the 5 captions are passed
    # For prediction, any of the captions may be used.

    dset_middle_train_p = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/dset_middle_train_p.npy')
    dset_middle_val_p = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/dset_middle_val_p.npy')

    for i in range(0, len(dset_middle_empty_train)):
        dset_middle_empty_train[i][start:end, start:end, :] = dset_middle_train_p[i] * 2.0 - 1
        # dset_middle_empty_train[i][start:end, start:end, :] = np.random.rand(32,32,3)*2.0 - 1
    for i in range(0, len(dset_middle_empty_val)):
        dset_middle_empty_val[i][start:end, start:end, :] = dset_middle_val_p[i] * 2.0 - 1
        # dset_middle_empty_val[i][start:end, start:end, :] = np.random.rand(32, 32, 3) * 2.0 - 1

    print "DELETE SCRAP DATASETS"
    del (dset_middle_train_p)
    del (dset_middle_val_p)

    print "##################"
    print "GET CAPTIONS"
    print "##################"
    train_captions = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/all_words_2_vectors_train.npy',
                             mmap_mode='r+')
    val_captions = np.load('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/all_words_2_vectors_val.npy',
                           mmap_mode='r+')

    print "GET GAN MODEL"
    gan_merged_model, adversary_model = model_zoo.DC_caption_LSTM_inception_exact(shape_img,
                                                                                  [128, 256, 512, 1024],
                                                                                  noise=True)

    losses = {"d": [], "g": [], "m": []}
    losses_per_frq = {"d": [], "g": [], "m": []}

    sgd = model_zoo.SGD(lr=0.0002, momentum=0.5, nesterov=True)
    adam = model_zoo.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9, epsilon=1e-8, decay=0.0)

    print "COMPILE GAN MERGED MODEL"
    # The loss is reduced so as to not completely constrain the filters to overfit to the training data
    model_zoo.alpha = 0.2 / 10  # SSIM and MSE loss
    model_zoo.beta = 0.8 / 10  # MSE loss kept higher to avoid patchy reconstructions

    gan_merged_model.compile(optimizer=adam, loss=model_zoo.loss_DSSIM_theano)
    gan_merged_model.summary()
    print "####################################"

    print "COMPILE ADVERSARY MODEL"
    adversary_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model.summary()
    print "####################################"

    print "COMPILE FULL MODEL"
    full_model_tensor = adversary_model(gan_merged_model.output)
    full_model = model_zoo.Model(outputs=full_model_tensor, inputs=gan_merged_model.input)
    full_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    full_model.summary()
    model_weights = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/full_model_weights.h5'
    print "####################################"

    dset_max = np.vstack((dset_train, dset_middle_empty_train))
    dset_real_false = np.zeros(((dset_middle_empty_train.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model, True)
    adversary_model.summary()

    print "#########################"
    print "Start ADVERSARY Fit INITIAL"
    print "#########################"

    adversary_model.fit(dset_max, dset_real_false,
                        batch_size=100,
                        nb_epoch=n_epoch,
                        shuffle=True,
                        verbose=True)

    del (dset_max)
    del (dset_real_false)

    print "START SLOW CONVERGE BATCH TRAINING"

    for i in range(0, num_runs):

        print "###########################"
        print i + 1
        print "###########################"
        cp = 0.15 - 0.01 * (i)

        if (cp < 0.0):
            cp = 0.0

        train_for_n_captions(dset_train,
                             dset_middle_empty_train,
                             train_captions,
                             adversary_model,
                             gan_merged_model,
                             full_model,
                             losses,
                             losses_per_frq,
                             nb_epoch=6000,
                             plt_frq=100,
                             batch_size=64,
                             predict_frq=1000,
                             run_num=i, m_train=True,
                             cp=cp)



        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        captions = np.zeros((train_captions.shape[0], train_captions.shape[2], train_captions.shape[3]))
        for j in range(0, len(train_captions)):
            captions[j] = train_captions[j, np.random.randint(0, train_captions.shape[1], 1)]

        predictions = gan_merged_model.predict([dset_middle_empty_train, captions, dset_middle_empty_train],
                                               batch_size=100)

        # Strengthen adversary to stabilize.
        strengthen_adversary(dset_train, predictions, adversary_model, n_epoch=round(cp + 0.01(i)))

        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp07_dcgan_captions_run_train' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"
        captions = np.zeros((val_captions.shape[0], val_captions.shape[2], val_captions.shape[3]))
        for j in range(0, len(val_captions)):
            captions[j] = val_captions[j, np.random.randint(0, val_captions.shape[1], 1)]
        predictions = gan_merged_model.predict([dset_middle_empty_val, captions, dset_middle_empty_val], batch_size=100)


        print "SAVE IMAGES"
        write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp07_dcgan_captions_run_test' + \
                     str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

        del (captions)
        del (predictions)
        full_model.save_weights(model_weights)
        full_model.save('/usr/local/data/sejacob/lifeworld/data/inpainting/models/caption_full_model_inception.h5')

    print "FINISHED ALL RUNS"


def strengthen_adversary(dset_train, predictions, adversary_model, n_epoch=1):
    dset_max = np.vstack((dset_train, predictions))
    dset_real_false = np.zeros(((predictions.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model, True)

    print "#########################"
    print "Start ADVERSARY WORKOUT Fit "
    print "#########################"

    adversary_model.fit(dset_max, dset_real_false,
                        batch_size=100,
                        nb_epoch=n_epoch,
                        shuffle=True,
                        verbose=True)
