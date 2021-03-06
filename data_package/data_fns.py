import cPickle as pkl
from tempfile import mkdtemp

import cv2
import gensim
import numpy as np
import os
from sklearn import decomposition


def pca_on_vectors(path, n_rows, train_or_val='train'):
    load_path = path + train_or_val + '_vectors_dset.npy'
    dset = np.load(load_path)
    dset_pca = np.zeros((dset.shape[0], n_rows, dset.shape[2]))

    for i in range(0, dset_pca.shape[0]):
        pca = decomposition.PCA(n_components=10)
        pca.fit(dset[i].transpose())
        x = pca.transform(dset[i].transpose()).transpose()
        dset_pca[i] = x

    save_path = path + train_or_val + '_pca_vectors_dset.npy'
    np.save(save_path, dset_pca)


def make_word_2_vec(path, save_path, train_or_val='train'):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/usr/local/data/sejacob/lifeworld/data/inpainting/word2vec/GoogleNews-vectors-negative300.bin',
        binary=True)
    with open('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/unique_words.pkl', 'rb') as f:
        dict = pkl.load(f)

    save_path = save_path + train_or_val + '_vectors_dset.npy'
    fname = os.path.join(mkdtemp(), 'newfile.dat')
    vectors = np.zeros((50, 300))

    filelist = sorted(os.listdir(path))

    dset = np.memmap(fname, dtype='float32', mode='w+',
                     shape=(len(filelist), vectors.shape[0], vectors.shape[1]))

    for i in range(0, len(filelist)):
        key = filelist[i][:-4]
        vectors = np.zeros((50, 300))
        k = 0
        for j in dict[key]:
            try:
                vector = model[j]
            except KeyError:
                continue
            vectors[k] = vector
            k += 1

        dset[i] = vectors

    np.save(save_path, dset)
    print ("Save dataset finished")


def save_dataset(path, save_path, train_or_val='train'):
    filelist = sorted(os.listdir(path))
    img = cv2.imread(path + filelist[0], cv2.IMREAD_COLOR)
    shape_img = img.shape
    save_path = save_path + train_or_val + '_dset.npy'
    fname = os.path.join(mkdtemp(), 'newfile.dat')
    dset = np.memmap(fname, dtype='float32', mode='w+',
                     shape=(len(filelist), shape_img[0], shape_img[1], shape_img[2]))

    for i in range(0, len(filelist)):
        filename = path + filelist[i]
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        norm_image = cv2.normalize(image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dset[i] = norm_image

    np.save(save_path, dset)
    print ("Save dataset finished")
    return True


def load_dataset(load_path):
    dset = np.load(load_path, mmap_mode='r+')
    return dset


def make_input_output_set(save_path, train_or_val='train'):
    dset_path = save_path + train_or_val + '_dset.npy'
    dset = load_dataset(dset_path)
    img = dset[0]
    assert (img.shape[0] == img.shape[1])
    len_side = img.shape[0]
    start = int(round(len_side / 4))
    end = int(round(len_side * 3 / 4))
    shape_middle = (dset.shape[0], int(dset.shape[1] / 2), int(dset.shape[2] / 2), dset.shape[3])

    save_path_middle = save_path + train_or_val + '_dset_middle.npy'
    save_path_middle_empty = save_path + train_or_val + '_dset_middle_empty.npy'

    fname1 = os.path.join(mkdtemp(), 'newfile2.dat')
    fname2 = os.path.join(mkdtemp(), 'newfile3.dat')

    dset_middle = np.memmap(fname1, dtype='float32', mode='w+', shape=shape_middle)
    dset_empty_middle = np.memmap(fname2, dtype='float32', mode='w+', shape=dset.shape)

    for i in range(0, len(dset)):
        dset_empty_middle[i] = dset[i]
        img = dset[i]
        middle_of_img = img[start:end, start:end, :]
        dset_empty_middle[i][start:end, start:end, :] = 0.0
        dset_middle[i] = middle_of_img

    np.save(save_path_middle, dset_middle)
    np.save(save_path_middle_empty, dset_empty_middle)
    print ("Make I/O set finished")
    return True


def get_datasets(save_path, train_or_val='train'):
    dset_path = save_path + train_or_val + '_dset.npy'
    save_path_middle = save_path + train_or_val + '_dset_middle.npy'
    save_path_middle_empty = save_path + train_or_val + '_dset_middle_empty.npy'

    dset = np.load(dset_path, mmap_mode='r+')
    dset_middle = np.load(save_path_middle, mmap_mode='r+')
    dset_middle_empty = np.load(save_path_middle_empty, mmap_mode='r+')

    return dset, dset_middle, dset_middle_empty


def write_predicted(test_set, test_set_middle_empty, predictions, write_path, middle_only=True):
    rows = test_set[0].shape[0]
    cols = test_set[0].shape[1]
    channels = test_set[0].shape[2]

    assert (rows == cols)

    image = np.zeros((rows, cols * 3 + 20, channels))
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    assert (len(test_set) == len(test_set_middle_empty) == len(predictions))

    for i in range(0, len(test_set)):
        image = np.zeros((rows, cols * 3 + 20, channels))
        image[:, 0:cols, :] = test_set[i]
        test_set_middle_empty[i][start:end, start:end, :] = 0.0
        image[:, cols + 10:cols * 2 + 10, :] = test_set_middle_empty[i]
        if (middle_only):
            middle_filled_image = test_set_middle_empty[i]
            middle_filled_image[start:end, start:end, :] = predictions[i]
            image[:, cols * 2 + 20:cols * 3 + 20, :] = middle_filled_image
        else:
            image[:, cols * 2 + 20:cols * 3 + 20, :] = predictions[i]

        # filename
        filename = write_path + str(i).zfill(len(str(len(test_set)))) + '.jpg'
        # imwrite
        image = ((image + 1) * (255 / 2)).astype('uint8')
        cv2.imwrite(filename, image)
