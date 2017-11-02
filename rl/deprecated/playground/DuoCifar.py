from keras import Input
from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import keras
import numpy as np
from keras.utils import plot_model

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

y2_train = np.array([np.array([1 if x == y_train[y][0] else 0 for x in range(10)]) for y in range(len(y_train))])
y2_test = np.array([np.array([1 if x == y_test[y][0] else 0 for x in range(10)]) for y in range(len(y_test))])


# Convolution Stream 1
input_img = Input(shape=x_train.shape[1:], name='image_input')
conv_1 = Conv2D(32, (3, 3), padding='same', input_shape=(1, 32, 32), activation='relu')(input_img)
conv_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_1)
max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)
dropout_1 = Dropout(0.25)(max_pool_1)
conv_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(dropout_1)
conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_3)
max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)
dropout_2 = Dropout(0.25)(max_pool_2)
conv_flatten = Flatten()(dropout_2)

# Convolution Stream 2
_input_img = Input(shape=x_train.shape[1:], name='image_inputs')
_conv_1 = Conv2D(32, (3, 3), padding='same', input_shape=(1, 32, 32), activation='relu')(_input_img)
_conv_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(_conv_1)
_max_pool_1 = MaxPooling2D(pool_size=(2, 2))(_conv_2)
_dropout_1 = Dropout(0.25)(_max_pool_1)
_conv_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(_dropout_1)
_conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(_conv_3)
_max_pool_2 = MaxPooling2D(pool_size=(2, 2))(_conv_4)
_dropout_2 = Dropout(0.25)(_max_pool_2)
_conv_flatten = Flatten()(_dropout_2)


# Concatinate
concat_layer = keras.layers.concatenate([conv_flatten, _conv_flatten])
dense_2 = Dense(512, activation='relu')(concat_layer)
dense_3 = Dense(10, activation='relu', name='class_output')(dense_2)

model = Model(inputs=[input_img, _input_img], outputs=[dense_3])
optimizer = keras.optimizers.adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model, to_file='./model.png')
print("Model created!")
model.fit({
    'image_inputs': x_train,
    'image_input': x_train,
    'class_output': y_train
}, y_train, batch_size=32, epochs=50, shuffle=True)




