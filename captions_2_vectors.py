import cPickle as pkl
import string
from tempfile import mkdtemp
import os
import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format(
    '/usr/local/data/sejacob/lifeworld/data/inpainting/word2vec/GoogleNews-vectors-negative300.bin',
    binary=True)

with open('/usr/local/data/sejacob/lifeworld/data/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl') as f:
    dict = pkl.load(f)

keys = dict.keys()

max = 0

for i in range(0, len(dict)):

    for j in range(0, len(dict[keys[i]])):
        sen = dict[keys[i]][j]
        # Remove punctuation
        sen = sen.translate(None, string.punctuation)
        dict[keys[i]][j] = sen

        len_sen = len(sen.split())
        if (len_sen > max):
            max = len_sen

print "SAVE TRAIN STUFF"
path = '/usr/local/data/sejacob/lifeworld/data/inpainting/train2014/'
filelist = sorted(os.listdir(path))

print max + 1

fname = os.path.join(mkdtemp(), 'newfile.dat')
dset_train = np.memmap(fname, dtype='float32', mode='w+', shape=(len(filelist), 5, max + 1, 300))

for i in range(0, len(filelist)):
    key = filelist[i][:-4]
    for j in range(0, len(dict[key])):
        if (j > 4):
            continue
        l = dict[key][j].split()
        m = 0
        for k in range(0, len(l)):
            # print l[k]
            try:
                vector = model[l[k]]
            except KeyError:
                continue
            dset_train[i, j, m, :] = vector
            m += 1

np.save('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/all_words_2_vectors_train.npy', dset_train)

print "SAVE VAL STUFF"
path = '/usr/local/data/sejacob/lifeworld/data/inpainting/val2014/'
filelist = sorted(os.listdir(path))

fname = os.path.join(mkdtemp(), 'newfile.dat')
dset_val = np.memmap(fname, dtype='float32', mode='w+', shape=(len(filelist), 5, max + 1, 300))

for i in range(0, len(filelist)):
    key = filelist[i][:-4]
    for j in range(0, len(dict[key])):
        if (j > 4):
            continue
        l = dict[key][j].split()
        m = 0
        for k in range(0, len(l)):
            # print l[k]
            try:
                vector = model[l[k]]
            except KeyError:
                continue
            dset_train[i, j, m, :] = vector
            m += 1

np.save('/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/all_words_2_vectors_val.npy', dset_val)
