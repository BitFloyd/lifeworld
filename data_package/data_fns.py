import cv2
import os
import numpy as np
from tempfile import mkdtemp


def save_dataset(path, save_path, train_or_val='train'):
    filelist = sorted(os.listdir(path))
    img = cv2.imread(path + filelist[0], cv2.IMREAD_COLOR)
    shape_img = img.shape
    save_path = save_path + train_or_val + '_dset.npy'
    fname = os.path.join(mkdtemp(), 'newfile.dat')
    dset = np.memmap(fname, dtype='uint8', mode='w+',
                     shape=(len(filelist), shape_img[0], shape_img[1], shape_img[2]))

    for i in range(0,len(filelist)):
        filename = path + filelist[i]
        dset[i] = cv2.imread(filename, cv2.IMREAD_COLOR)


    np.save(save_path,dset)
    print ("Save dataset finished")
    return True


def load_dataset(load_path):
    dset = np.load(load_path,mmap_mode='r+')
    return dset


def make_input_output_set(save_path, train_or_val='train'):
    dset_path = save_path + train_or_val + '_dset.npy'
    dset = load_dataset(dset_path,)
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

    dset_middle = np.memmap(fname1, dtype='uint8', mode='w+', shape=shape_middle)
    dset_empty_middle = np.memmap(fname2, dtype='uint8', mode='w+', shape=dset.shape)

    for i in range(0, len(dset)):
        dset_empty_middle[i] = dset[i]
        img = dset[i]
        middle_of_img = img[start:end, start:end, :]
        dset_empty_middle[i][start:end, start:end, :] = 0
        dset_middle[i] = middle_of_img

    np.save(save_path_middle,dset_middle)
    np.save(save_path_middle_empty,dset_empty_middle)
    print ("Make I/O set finished")
    return True


def get_datasets(save_path, train_or_val='train'):
    dset_path = save_path + train_or_val +  '_dset.npy'
    save_path_middle = save_path + train_or_val + '_dset_middle.npy'
    save_path_middle_empty = save_path + train_or_val + '_dset_middle_empty.npy'

    dset = np.load(dset_path,mmap_mode='r+')
    dset_middle = np.load(save_path_middle,mmap_mode='r+')
    dset_middle_empty = np.load(save_path_middle_empty,mmap_mode='r+')

    return dset, dset_middle, dset_middle_empty
