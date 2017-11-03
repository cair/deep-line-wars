import numpy as np
from PIL import Image
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.contrib.keras.python.keras.layers.merge import Concatenate
from tensorflow.contrib.keras.python.keras.layers.noise import GaussianNoise
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop
from tensorflow.contrib.keras.python.keras.engine import Input, Model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Activation, Reshape, Dropout
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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
        http://wiseodd.github.io/techblog/2016/12/24/conditional-gan-tensorflow/


        :param memory:
        :param graph:
        :param batch_size:
        :param state_size:
        :param q_model:
        """

        self.latent_size = 100
        self.action_size = 13
        self.memory = memory
        self.graph = graph
        self.q_model = q_model
        self.BATCH_SIZE = batch_size
        self.state_size = state_size

        # Adversarial Parameters
        self.a_lr = 0.0001
        self.a_decay = 3e-8

        # Generator Parameters
        #self.g_lr = 0.0008
        #self.g_decay = 6e-8
        self.g_dropout = 0.4
        self.g_momentum = 0.9

        # Discriminator Parameters
        self.d_lr = 0.0002
        self.d_decay = 6e-8
        self.d_dropout = 0.4
        self.d_depth = 64

        self.generator_model = self.build_generator_model()
        self.discriminator_model = self.build_discriminator_model()
        self.adversarial_model = self.build_adversarial(self.generator_model, self.discriminator_model)

        # Statistics
        self.stat_real_q = []
        self.stat_fake_q = []
        self.d_loss = []
        self.a_loss = []
        self.q_loss = []
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.d_loss_plot = self.figure.add_subplot(2, 2, 1)
        self.q_loss_plot = self.figure.add_subplot(2, 2, 2)
        self.a_loss_plot = self.figure.add_subplot(2, 2, 3)
        #self.plot_q = self.figure.add_subplot(2, 2, 4)
        self.canvas = FigureCanvasAgg(self.figure)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()

    def dqncdcgan_train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        experience_replay = self.memory.get(int(self.BATCH_SIZE / 2))
        fake_q = np.array([np.random.uniform(-1.0, 1.0, (self.action_size, )) for _ in range(self.BATCH_SIZE)])
        real_q = np.array([q_values for state, q_values in experience_replay])

        noise = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, self.latent_size])
        q_values = np.concatenate((fake_q, real_q))

        generated_data = self.generator_model.predict([noise, q_values])
        fake_s = generated_data[0]
        fake_q = generated_data[1]

        self.stitch_generated_image(fake_s, "real_fake_q")

        # TODO
        #loss = self.q_model.train_on_batch(fake_s)
        #print(loss)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        self.dqncdcgan_train()

        with self.graph.as_default():
            experience_replay = self.memory.get(self.BATCH_SIZE)
            real_s = np.array([state[0] for state, q_values in experience_replay])
            real_q = np.array([q_values for state, q_values in experience_replay])

            # Generate fake state and q
            noise = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, self.latent_size])

            generated_data = self.generator_model.predict([noise, real_q])
            fake_s = generated_data[0]
            fake_q = generated_data[1]

            self.stitch_generated_image(real_s, "real")
            self.stitch_generated_image(fake_s, "fake")

            # Concatenate fake and real images
            x_0 = np.concatenate((real_s, fake_s))
            x_1 = np.concatenate((real_q, fake_q))

            y = np.ones([2 * self.BATCH_SIZE, 1])   # First half of image-set is REAL (0)
            y[self.BATCH_SIZE:, :] = 0  # Last half of image-set is FAKE (0)

            d_loss = self.discriminator_model.train_on_batch([x_0, x_1], y)

            # Generate new noise
            y = np.ones([self.BATCH_SIZE, 1])   # Set all images class to REAL
            noise = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, self.latent_size])

            a_loss = self.adversarial_model.train_on_batch([noise, real_q], y)

            # Compare Q values
            new_real_q_values = self.q_model.predict(real_s)
            fake_q_values = self.q_model.predict(fake_s)

            self.stat_fake_q = fake_q_values
            self.stat_real_q = new_real_q_values

            self.create_plot()

            q_loss_mse = ((fake_q_values - new_real_q_values) ** 2).mean(axis=None)

            self.a_loss.append(a_loss[0])
            self.d_loss.append(d_loss[0])
            self.q_loss.append(q_loss_mse)

    def build_adversarial(self, generator, discriminator):

        model = Model(inputs=generator.input, outputs=[discriminator(generator.output)])

        optimizer = RMSprop(lr=self.a_lr, decay=self.a_decay)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, to_file='./output/_adversarial_model.png', show_shapes=True, show_layer_names=True)
        return model

    def dqn_generator_bridge(self):

        q_input = Input(shape=(self.action_size, ))

        x = Dense(100)(q_input)
        noise = GaussianNoise(0.01)(x)
        noise = Dense(100)(noise)

        model = Model(inputs=[q_input], outputs=[noise, q_input])
        plot_model(model, to_file='./output/_dqn_generator_bridge_model.png', show_shapes=True, show_layer_names=True)
        return model

    def build_discriminator_model(self):

        image_input = Input(shape=(84, 84, 1))
        q_input = Input(shape=(13, ))

        x = Conv2D(self.d_depth, (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(image_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 8, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Reshape((int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))(x)
        x = Flatten()(x)

        x = Concatenate()([q_input, x])

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        model = Model(inputs=[image_input, q_input], outputs=[x])
        model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(lr=self.d_lr, clipvalue=1.0, decay=self.d_decay),
            metrics=['accuracy']
        )
        plot_model(model, to_file='./output/_discriminator_model.png', show_shapes=True, show_layer_names=True)

        return model

    def build_generator_model(self):
        depth = 64 + 64 + 64 + 64
        dim = 21

        latent_input = Input(shape=(self.latent_size, ))
        q_input = Input(shape=(self.action_size, ))
        x = Concatenate()([latent_input, q_input])

        x = Dense(1014)(x)
        x = Dense(depth * dim * dim)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Reshape((dim, dim, depth))(x)
        x = Dropout(self.g_dropout)(x)

        # Upsample 42
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(int(depth / 2), 5, padding='same')(x)
        x = BatchNormalization(momentum=self.g_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Upsample 84
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(int(depth / 4), 5, padding='same')(x)
        x = BatchNormalization(momentum=self.g_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(int(depth / 8), 5, padding='same')(x)
        x = BatchNormalization(momentum=self.g_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(1, 5, padding='same')(x)
        x = Activation('sigmoid')(x)

        model = Model(inputs=[latent_input, q_input], outputs=[x, q_input])
        #model.compile(optimizer=Adam(lr=self.g_lr, clipvalue=1.0, decay=self.g_decay), loss="mse")
        plot_model(model, to_file='./output/_generator_model.png', show_shapes=True, show_layer_names=True)

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
        name = 'dqn_gan_%s.png' % name
        img.save('./output/tmp_' + name)
        os.rename('./output/tmp_' + name, './output/' + name)

    def create_plot(self):
        x = np.arange(len(self.q_loss))

        self.d_loss_plot.cla()
        self.a_loss_plot.cla()
        self.q_loss_plot.cla()

        self.d_loss_plot.plot(x, self.d_loss, label="D-Loss")
        self.a_loss_plot.plot(x, self.a_loss, label="A-Loss")
        self.q_loss_plot.plot(x, self.q_loss, label="Q-Loss")

        self.d_loss_plot.set_ylabel('Value')
        self.d_loss_plot.set_title('Discriminator Loss')

        self.a_loss_plot.set_ylabel('Value')
        self.a_loss_plot.set_title('Adversarial Loss')

        self.q_loss_plot.set_ylabel('Value')
        self.q_loss_plot.set_title('Q Loss')

        #plot_action.bar(np.arange(self.action_size), self.stat_fake_q, align='center', alpha=0.5)
        #plot_action.bar(np.arange(self.action_size), self.stat_real_q, align='center', alpha=0.5)

        self.figure.savefig('./output/tmp_dqn_gan_plot.png')
        os.rename('./output/tmp_dqn_gan_plot.png', './output/dqn_gan_plot.png')