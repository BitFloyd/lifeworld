from keras.models import load_model

from model_pkg import visualizations

# Get data as numpy mem-map to not overload the RAM
print "GET MODEL"
model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg_mp_False.model'
model = load_model(model_filepath)

model_vis_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp01_vgg_mp_False.model.png'

visualizations.make_model_visualization(model, model_vis_filepath)
