from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine import Input
from tensorflow.contrib.keras.python.keras.engine.training import Model
from tensorflow.contrib.keras.python.keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Activation, Reshape, Dropout, Flatten, Lambda
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model


m = 0.9
lr = 0.0008
dropout = 0.4
depth = 64 + 64 + 64 + 64
_dim = 21

# Generator
g_input = Input(batch_shape=(None, 100))

x = Dense(_dim * _dim * depth, input_dim=100)(g_input)
x = BatchNormalization(momentum=0.9)(x)
x = Activation('relu')(x)
x = Reshape((_dim, _dim, depth))(x)
x = Dropout(dropout)(x)

x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(int(depth/2), 5, padding='same')(x)
x = BatchNormalization(momentum=m)(x)
x = Activation('relu')(x)

x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(int(depth/4), 5, padding='same')(x)
x = BatchNormalization(momentum=m)(x)
x = Activation('relu')(x)

x = Conv2DTranspose(int(depth/8), 5, padding='same')(x)
x = BatchNormalization(momentum=m)(x)
x = Activation('relu')(x)


x = Conv2DTranspose(1, 5, padding='same')(x)
x = Activation('sigmoid')(x)

g_model = Model(inputs=[g_input], outputs=[x])
g_model.compile(optimizer=RMSprop(lr=lr, clipvalue=1.0, decay=6e-8), loss="mse")



# Discriminator
input_layer = Input(shape=(84, 84, 1), name='image_input')
x = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(input_layer)
x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(x)
x = Reshape((20, 20, 32))(x)
x = Flatten()(x)

vs = Dense(512, activation="relu", kernel_initializer='uniform')(x)
vs = Dense(1, kernel_initializer='uniform')(vs)

ad = Dense(512, activation="relu", kernel_initializer='uniform')(x)
ad = Dense(13, kernel_initializer='uniform')(ad)

policy = Lambda(lambda w: w[0] - K.mean(w[0]) + w[1])([vs, ad])
d_model = Model(inputs=[input_layer], outputs=[policy])
d_model.compile(optimizer=RMSprop(lr=0.0008), loss="mse")


# Adverserial
# ERROR IS HERE
connected = d_model(g_model.output)
model = Model(inputs=[g_model.input], outputs=[connected])
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # TODO loss mse?
plot_model(model, to_file='adversarial_model.png', show_shapes=True, show_layer_names=True)

model.summary()