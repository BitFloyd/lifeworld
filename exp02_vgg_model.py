from data_package import data_fns
from model_pkg import model_zoo
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from keras.models import load_model

# Get data as numpy mem-map to not overload the RAM
print "GET_DATASETS"
save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
shape_img = dset_train[0].shape

# EXPERIMENT 1
# --------------
# VGG type model. Max pool in the middle of the representation to reduce
# the size in half.
# If reconstruction doesnt work well enough, try max-pool at the end as well.
# Key-point: No max-pooling anywhere else because.... translation independence not important.

print "GET MODEL"
model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp02_vgg.model'

es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)
checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000001, verbose=1)

model = model_zoo.vgg_model_non_linear(shape_img, maxpool=True, op_only_middle=True,highcap=False)

#if (os.path.isfile(model_filepath)):
#    model = load_model(model_filepath)

print "START FIT"
history = model.fit(dset_middle_empty_train, dset_middle_train,
                    batch_size=200, nb_epoch=200,
                    callbacks=[es, checkpoint],
                    validation_split=0.3,
                    shuffle=True)

print "MAKE PREDICTIONS"
predictions = model.predict(dset_middle_empty_val, batch_size=100)


print "SAVE IMAGES"
write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp02_maxpool_True_highcap_False_mp_middle/'
data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path)