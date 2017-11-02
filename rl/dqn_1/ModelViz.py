
import time
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from scipy.misc import imsave
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.models import load_model
from multiprocessing import Pool


class ModelViz:

    def __init__(self):
        pass

    def normalize(self, x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        #if x.shape[2] == 3:
        #    x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def process(self, model_name, model=None, conv_layer_idx=0, file_id=int(time.time())):
        with tf.device('/cpu:0'):
            if model is not None:
                model = model
            else:
                model = load_model(model_name)


            # Find last conv layer
            conv_layers = [layer for layer in model.layers if type(layer) == Conv2D]
            conv_layer = conv_layers[conv_layer_idx]
            filters = conv_layer.output_shape[3]

            input_img = model.input
            tensor_shape = input_img.get_shape()

            img_width = int(tensor_shape[1])
            img_height = int(tensor_shape[2])
            img_channels = int(tensor_shape[3])

            kept_filters = []

            print("Processing %s with filters: %s" % (conv_layer.name, filters))

            for filter_index in range(0, filters):
                #print('Processing filter %d' % filter_index)


                layer_output = conv_layer.output
                if K.image_data_format() == 'channels_first':
                    loss = K.mean(layer_output[:, filter_index, :, :])
                else:
                    loss = K.mean(layer_output[:, :, :, filter_index])


                # we compute the gradient of the input picture wrt this loss
                grads = K.gradients(loss, input_img)[0]

                # normalization trick: we normalize the gradient
                grads = self.normalize(grads)

                # this function returns the loss and grads given the input picture
                iterate = K.function([input_img], [loss, grads])

                # step size for gradient ascent
                step = 1.

                # we start from a gray image with some random noise
                if K.image_data_format() == 'channels_first':
                    input_img_data = np.random.random((1, img_channels, img_width, img_height))
                else:
                    input_img_data = np.random.random((1, img_width, img_height, img_channels))
                input_img_data = (input_img_data - 0.5) * 20 + 128

                # we run gradient ascent for 20 steps
                for i in range(40):
                    loss_value, grads_value = iterate([input_img_data])
                    input_img_data += grads_value * step

                    #print('Current loss value:', loss_value)
                    if loss_value <= 0.:
                        # some filters get stuck to 0, we can skip them
                        break

                # decode the resulting input image
                if loss_value > 0:

                    img = self.deprocess_image(input_img_data[0])
                    kept_filters.append((img, loss_value))
                end_time = time.time()
               # print('Filter %d processed' % filter_index)



            # we will stich the best 64 filters on a 8 x 8 grid.
            n = 4

            # the filters that have the highest loss are assumed to be better-looking.
            # we will only keep the top 64 filters.
            kept_filters.sort(key=lambda x: x[1], reverse=True)
            kept_filters = kept_filters[:n * n]

            # build a black picture with enough space for
            # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
            margin = 5

            width = n * img_width + (n - 1) * margin
            height = n * img_height + (n - 1) * margin
            stitched_filters = np.zeros((width, height, img_channels))


            # fill the picture with our saved filters
            for i in range(n):
                for j in range(n):
                    try:
                        img, loss = kept_filters[i * n + j]

                        print(img)

                        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                        (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
                    except:
                        pass
            # save the result to disk

            img = Image.fromarray(stitched_filters[:, :, 0])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save('./output/dqn_filter_%s_%s_%s.png' % (conv_layer.name, filters, file_id))
            #imsave('./output/dqn_filter_%s_%s_%s.png' % (conv_layer.name, filters, file_id), stitched_filters)


def start(file_paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    m = ModelViz()
    for file_path in file_paths:
        file_id = file_path.split("_")[2].split(".")[0]

        model = load_model(file_path)
        n_conv_layers = len([layer for layer in model.layers if type(layer) == Conv2D])
        for i in range(n_conv_layers):
            m.process(file_path, conv_layer_idx=i, file_id=file_id)

def chunkify(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":

    n_processes = 16
    save_path = "./save"
    files = [os.path.join(save_path, file) for file in list(sorted(os.listdir(save_path)))]
    chunks = list(chunkify(files, n_processes))

    with Pool(n_processes) as p:
        print(p.map(start, chunks))

    """m = ModelViz()

    save_path = "./save"
    for file in list(sorted(os.listdir(save_path)))[-1:]:
        file_path = os.path.join(save_path, file)
        for i in range(0, 3):
            file_id = file.split("_")[2].split(".")[0]
            m.process(file_path, conv_layer_idx=i, file_id=file_id)"""
