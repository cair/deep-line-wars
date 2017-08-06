import threading

import time
import numpy as np
import matplotlib.pyplot as plt

class ModelViz(threading.Thread):

    def __init__(self, algorithm):
        threading.Thread.__init__(self)
        self.algorithm = algorithm

    def run(self):
        self.dump()

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
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def dump(self):
        while True:
            if self.algorithm.target_model:
                inp_l = self.algorithm.target_model.layers[0]
                conv_0 = self.algorithm.target_model.layers[1]
                conv_1 = self.algorithm.target_model.layers[2]
                conv_2 = self.algorithm.target_model.layers[3]

                layers = [conv_0, conv_1, conv_2]

                w = conv_0.get_weights()[0]
                w = np.squeeze(w)
                plt.figure(figsize=(15, 15))
                plt.title('conv1 weights')
                #nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)


                print(w.shape)


            time.sleep(10)
