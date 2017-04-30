import os
import imageio

batch_size = 40
root_name = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/dcgan_mcomplex_strong_disc/'
dst_folder = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/training_gifs/'
dirlist = os.listdir(root_name)
order_list = []
os.chdir(root_name)
dirs = filter(os.path.isdir, os.listdir(root_name))
# files = [os.path.join(search_dir, f) for f in files] # add path to each file
dirs.sort(key=lambda x: os.path.getmtime(x))
print dirs
# matching = [s for s in dirlist if "exp07" in s]
# matching_batch = [s for s in matching if "batch01" in s]
# matching_batch_iter = [s for s in matching_batch if "_99" in s]
# j = 0
#
# while matching_batch:
#     i = 0
#     while matching_batch_iter:
#         matching_batch_iter = [s for s in matching_batch if "_" + str(100 * i + 99) in s]
#         print matching_batch_iter
#         order_list.append(matching_batch_iter)
#         i+=1
#
#     matching_batch = [s for s in matching if "batch"+str(j).zfill(len(str(batch_size))) in s]
#     j+=1
#
# order_list = order_list[0:-1]
# print order_list
#
for i in range(0, batch_size):
    images = []
    with imageio.get_writer(dst_folder + str(i).zfill(len(str(batch_size))) + ".gif", mode='I',
                            duration=3.0 / (len(dirs))) as writer:
        for j in range(0, len(dirs)):
            filename = root_name + dirs[j] + '/' + str(i).zfill(len(str(batch_size))) + ".jpg"

            image = imageio.imread(filename)

            if j == 0:
                for k in range(0, 10):
                    writer.append_data(image)
            elif (j == len(dirs) - 1):
                for k in range(0, 10):
                    writer.append_data(image)
            else:
                writer.append_data(image)
