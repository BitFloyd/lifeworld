from model_pkg import model_zoo
from model_pkg import visualizations

# Get data as numpy mem-map to not overload the RAM
print "GET MODELS"

gan_merged_model, adversary_model = model_zoo.DCGAN_model((64, 64, 3), [128, 256, 512, 1024], noise=True)

full_model_tensor = adversary_model(gan_merged_model.output)
full_model = model_zoo.Model(gan_merged_model.input, full_model_tensor)
full_model.summary()

gmm_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp05_01_gan_merged_model.png'
adm_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp05_01_adversary_model.png'
full_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp05_01_full_model.png'

visualizations.make_model_visualization(gan_merged_model, gmm_filepath)
visualizations.make_model_visualization(adversary_model, adm_filepath)
visualizations.make_model_visualization(full_model, full_filepath)
