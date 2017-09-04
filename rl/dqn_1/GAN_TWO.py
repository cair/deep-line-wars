import numpy as np
from PIL import Image
from tensorflow.contrib.keras.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.contrib.keras.python.keras.layers.embeddings import Embedding
from tensorflow.contrib.keras.python.keras.layers.merge import Multiply
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling2D
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
        self.action_size = 13

        # Adversarial Parameters
        self.a_lr = 0.00004
        self.a_decay = 3e-16

        print(self.a_decay)

        # Generator Parameters
        self.g_lr = 0.0005
        self.g_decay = 6e-8
        self.g_dropout = 0.4
        self.g_momentum = 0.9
        self.latent_size = 100

        # Discriminator Parameters
        self.d_lr = 0.0005
        self.d_decay = 6e-8
        self.d_dropout = 0.3
        self.d_depth = 64

        self.generator_model = self.build_generator_model()
        self.discriminator_model = self.build_discriminator_model()
        self.adversarial_model = self.build_adversarial(self.generator_model, self.discriminator_model)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        with self.graph.as_default():
            experience_replay = self.memory.get(self.BATCH_SIZE)
            real_s = np.array([state[0] for state, q_values in experience_replay])
            real_q = np.array([q_values for state, q_values in experience_replay])

            # Generate fake state and q
            noise = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, self.latent_size])
            generated_data = self.generator_model.predict(noise)
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
            Y = np.ones([self.BATCH_SIZE, 1])   # Set all images class to REAL
            noise = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, self.latent_size])

            a_loss = self.adversarial_model.train_on_batch(noise, Y)


            # Compare Q values
            new_real_q_values = self.q_model.predict(real_s)
            fake_q_values = self.q_model.predict(fake_s)

            loss = np.mean(fake_q_values - new_real_q_values)

            print("D: ", d_loss, "A: ", a_loss, "Q_Loss: ", loss)

    def build_adversarial(self, generator, discriminator):
        model = Model(inputs=generator.input, outputs=discriminator(generator.output))

        optimizer = Adam(lr=self.a_lr, clipvalue=1.0, decay=self.a_decay)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, to_file='./output/_adversarial_model.png', show_shapes=True, show_layer_names=True)
        return model

    def build_discriminator_model(self):

        input_s = Input(shape=self.state_size)
        input_q = Input(shape=(self.action_size, ))

        x = Conv2D(self.d_depth, (3, 3), strides=(2, 2), padding='same')(input_s)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 2, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        """x = Conv2D(self.d_depth * 4, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)

        x = Conv2D(self.d_depth * 8, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.d_dropout)(x)"""

        x = Reshape((int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))(x)
        x = Flatten()(x)

        # Image Dense
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Q- Dense
        x_1 = Dense(1024)(input_q)
        x_1 = LeakyReLU(alpha=0.2)(x_1)
        x_1 = Dense(1)(x_1)
        x_1 = Activation('sigmoid')(x_1)


        # Output
        fake = Dense(1)(x)
        fake = Activation('sigmoid')(fake)
        fake = Multiply()([x_1, fake])

        model = Model(inputs=[input_s, input_q], outputs=[fake])

        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=self.d_lr, clipvalue=1.0, decay=self.d_decay,  nesterov=True),
            metrics=['accuracy']
        )
        plot_model(model, to_file='./output/_discriminator_model.png', show_shapes=True, show_layer_names=True)

        return model

    def build_generator_model(self):
        depth = 64 + 64 + 64 + 64

        latent = Input(shape=(self.latent_size, ))

        x = Dense(1014, input_dim=100)(latent)

        aux = Dense(self.action_size)(x)
        aux = Activation("linear")(aux)


        x = Activation('tanh')(x)


        x = Dense(128 * 21 * 21)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Reshape((21, 21, 128))(x)
        x = Dropout(self.g_dropout)(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(int(depth / 2), 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(int(depth / 4), 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(int(depth / 8), 5, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(1, 5, padding='same')(x)
        x = Activation('tanh')(x)


        """x = Dense(1024)(latent)
        x = LeakyReLU(alpha=0.2)(x)

        #x = Dense(128 * 21 * 21)(x)
        #x = LeakyReLU(alpha=0.2)(x)

        aux = Dense(self.action_size)(x)
        aux = Activation("linear")(aux)

        # Upsample to 42
        x = Reshape((21, 21, 256))(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Upsample to 84
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(1, (2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
        x = Activation('tanh')(x)"""

        model = Model(inputs=[latent], outputs=[x, aux])
        model.compile(optimizer=Adam(lr=self.g_lr, decay=self.g_decay, clipvalue=1.0), loss="mse")
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
        img.save('./output/dqn_gan_%s.png' % name)
