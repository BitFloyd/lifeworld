from data_package import data_fns

save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
path = '/usr/local/data/sejacob/lifeworld/data/inpainting/train2014/'
train_or_val = 'train'

data_fns.save_dataset(path,save_path,train_or_val)
data_fns.make_input_output_set(save_path,train_or_val)

path = '/usr/local/data/sejacob/lifeworld/data/inpainting/val2014/'
train_or_val='val'

data_fns.save_dataset(path,save_path,train_or_val)
data_fns.make_input_output_set(save_path,train_or_val)

