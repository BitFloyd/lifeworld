from keras.utils.visualize_util import plot


def make_model_visualization(model, filepath):
    plot(model, to_file=filepath, show_shapes=True, show_layer_names=True)
    return True
