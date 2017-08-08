import numpy as np
from PIL import Image
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.contrib.keras.python.keras.engine import Input, Model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model


class GAN:
    def __init__(self, memory, graph, batch_size, state_size, q_model):
        """
        Problem: generated images look like noise.
        Solution: use dropout on both Discriminator and Generator. Low dropout values (0.3 to 0.6) generate more realistic images.

        Problem: Discriminator loss converges rapidly to zero thus preventing the Generator from learning.
        Solution: Do not pre-train the Discriminator. Instead make its learning rate bigger than the Adversarial model learning rate. Use a different training noise sample for the Generator.

        Problem: generator images still look like noise.
        Solution: check if the activation, batch normalization and dropout are applied in the correct sequence.

        Problem: figuring out the correct training/model parameters.
        Solution: start with some known working values from published papers and codes and adjust one parameter at a time. Before training for 2000 or more steps, observe the effect of parameter value adjustment at about 500 or 1000 steps.
        Sample Outputs

        https://github.com/soumith/ganhacks

        :param memory:
        :param graph:
        :param batch_size:
        :param state_size:
        :param q_model:
        """


        self.memory = memory
        self.graph = graph
        self.q_model = q_model
        self.BATCH_SIZE = batch_size
        self.state_size = state_size

        # Adversarial Parameters
        self.a_lr = 0.00004
        self.a_decay = 3e-16

        print(self.a_decay)

        # Generator Parameters
        self.g_lr = 0.0008
        self.g_decay = 6e-8
        self.g_dropout = 0.4
        self.g_momentum = 0.9

        # Discriminator Parameters
        self.d_lr = 0.0008
        self.d_decay = 6e-8
        self.d_dropout = 0.4
        self.d_depth = 64

        self.generator_model = self.build_generator_model()
        self.discriminator_model = self.build_discriminator_model()
        self.adversarial_model = self.build_adversarial(self.generator_model, self.discriminator_model)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        with self.graph.as_default():
            real_experiences = self.memory.get(self.BATCH_SIZE)

            real_states = np.array([state[0] for state, q_values in real_experiences])
            real_q_values = np.array([q_values for state, q_values in real_experiences])
            noisy_q_values = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, 13])


            fake_states = self.generator_model.predict(noisy_q_values)

            self.stitch_generated_image(real_states, "real")
            self.stitch_generated_image(fake_states, "fake")

            X = np.concatenate((real_states, fake_states))
            Y = np.ones([2 * self.BATCH_SIZE, 1])
            Y[self.BATCH_SIZE:, :] = 0

            d_loss = self.discriminator_model.train_on_batch(X, Y)

            Y = np.ones([self.BATCH_SIZE, 1])
            noisy_q_values = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, 13])
            a_loss = self.adversarial_model.train_on_batch(noisy_q_values, Y)

            # Compare Q values
            new_real_q_values = self.q_model.predict(real_states)
            fake_q_values = self.q_model.predict(fake_states)

            loss = np.mean(fake_q_values - new_real_q_values)

            print("D: ", d_loss, "A: ", a_loss, "Q_Loss: ", loss)

    def build_adversarial(self, generator, discriminator):
        model = Model(inputs=[generator.input], outputs=[discriminator(generator.output)])

        optimizer = RMSprop(lr=self.a_lr, clipvalue=1.0, decay=self.a_decay)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, to_file='./output/adversarial_model.png', show_shapes=True, show_layer_names=True)
        return model

    def build_discriminator_model(self):

        input_ = Input(shape=self.state_size)
        x = Conv2D(self.d_depth, (3, 3), strides=(2, 2), padding='same')(input_)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 2, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 4, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 8, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Reshape((int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        model = Model(inputs=[input_], outputs=[x])
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=self.d_lr, clipvalue=1.0, decay=self.d_decay),
            metrics=['accuracy']
        )

        return model

    def build_generator_model(self):
        dim = np.array([84, 84, 1])
        depth = 64 + 64 + 64 + 64
        _dim = 21

        g_input = Input(shape=(13,))

        # Input: (100, )
        # Output:
        x = Dense(1014, input_dim=100)(g_input)
        #x = BatchNormalization(momentum=0.9)(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('tanh')(x)

        x = Reshape((_dim, _dim, depth))(x)
        x = Dropout(self.g_dropout)(x)

        # Input:
        # Output:
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(int(depth / 2), 5, padding='same')(x)
        #x = BatchNormalization(momentum=self.g_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Input:
        # Output:
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(int(depth / 4), 5, padding='same')(x)
        #x = BatchNormalization(momentum=self.g_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Input: 2dim[0] * 2dim[1] * dim[2]/8
        # Output: 28 x 28 x 1
        x = Conv2DTranspose(int(depth / 8), 5, padding='same')(x)
        #x = BatchNormalization(momentum=self.g_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 84 x 84 x 1
        x = Conv2DTranspose(1, 5, padding='same')(x)
        x = Activation('tanh')(x)

        model = Model(inputs=[g_input], outputs=[x])
        model.compile(optimizer=Adam(lr=self.g_lr, clipvalue=1.0, decay=self.g_decay), loss="mse")
        plot_model(model, to_file='./output/generator_model.png', show_shapes=True, show_layer_names=True)

        return model

    def stitch_generated_image(self, generated_images, name):
        # 4 x 4 Grid
        n = 4
        margin = 5
        i_w = self.state_size[0]
        i_h = self.state_size[1]
        i_c = self.state_size[2]
        width = n * i_w + (n - 1) * margin
        height = n * i_h + (n - 1) * margin
        stitched_filters = np.zeros((width, height, i_c))

        for i in range(n):
            for j in range(n):
                img = generated_images[i * n + j]

                stitched_filters[
                (i_w + margin) * i: (i_w + margin) * i + i_w,
                (i_h + margin) * j: (i_h + margin) * j + i_h, :] = img

        stitched_filters *= 255

        img = Image.fromarray(stitched_filters[:, :, 0])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save('./output/dqn_gan_%s.png' % name)
