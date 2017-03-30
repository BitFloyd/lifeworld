from model_pkg import model_zoo
from model_pkg import visualizations

print "GET MODELS"

gan_merged_model, adversary_model = model_zoo.DCGAN_ker2_caption_LSTM_inception_latent((64, 64, 3),
                                                                                       [128 / 2, 256 / 2, 512 / 2,
                                                                                        1024 / 2], noise=True)

full_model_tensor = adversary_model(gan_merged_model.output)
full_model = model_zoo.Model(gan_merged_model.input, full_model_tensor)
# full_model.summary()

gmm_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp07_gan_merged_model_latent.png'
adm_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp07_adversary_model_latent.png'
full_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp07_full_model_latent.png'

visualizations.make_model_visualization(gan_merged_model, gmm_filepath)
visualizations.make_model_visualization(adversary_model, adm_filepath)
visualizations.make_model_visualization(full_model, full_filepath)
