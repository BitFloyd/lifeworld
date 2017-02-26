from data_package import data_fns
from model_pkg import model_zoo
import os
from keras.models import load_model
from model_pkg import visualizations

# Get data as numpy mem-map to not overload the RAM
print "GET_DATASETS"
save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
shape_img = dset_train[0].shape

print "GET MODEL"
model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg.model'

model = model_zoo.vgg_model(shape_img, maxpool=True, op_only_middle=True,highcap=False)

if (os.path.isfile(model_filepath)):
    model = load_model(model_filepath)

model_vis_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg_model.png'

visualizations.make_model_visualization(model,model_vis_filepath)