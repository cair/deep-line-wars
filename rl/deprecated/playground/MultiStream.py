from keras import Input
from keras.datasets import cifar10, mnist
from keras.engine import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import keras
import matplotlib.pyplot as plt
from keras.utils import plot_model

DENSE_SIZE = 256
ONLY_CIFAR = False


# The data, shuffled and split between train and test sets:
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)
"""


# Convolution Stream 1
input_image = Input(shape=X_train.shape[1:], name='input_image')
conv_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_image)
dropout_1 = Dropout(0.25)(conv_1)


"""conv_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_1)
max_pool_1 = MaxPooling2D(pool_size=(1, 1))(conv_2)
dropout_1 = Dropout(0.25)(max_pool_1)
conv_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(dropout_1)
conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_3)
max_pool_2 = MaxPooling2D(pool_size=(1, 1))(conv_4)
dropout_2 = Dropout(0.25)(max_pool_2)"""

conv_flatten = Flatten()(conv_1)
conv_dense_1 = Dense(DENSE_SIZE)(conv_flatten)

# State Stream

input_state = Input(shape=Y_train[0].shape, dtype='float32', name='input_state')
state_dense_1 = Dense(DENSE_SIZE, activation='relu')(input_state)




if not ONLY_CIFAR:
    # Concatinate
    concat_layer = keras.layers.Concatenate()([state_dense_1, conv_dense_1])
    dense_2 = Dense(256, activation='relu')(concat_layer)
    dense_3 = Dense(10, activation='relu', name='output')(dense_2)
    model = Model(inputs=[input_image, input_state], outputs=[dense_3])
    optimizer = keras.optimizers.adam(lr=0.0001, decay=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='./model.png', show_shapes=True)

    history = model.fit({
        'input_state': Y_train,
        'input_image': X_train,
        'output': Y_train
    }, Y_train, batch_size=32, epochs=20, shuffle=True, validation_data=([X_test, Y_test], [Y_test]))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

else:
    conv_dense_1 = Dense(512, activation='relu')(conv_flatten)
    conv_dropout_1 = Dropout(0.5)(conv_dense_1)
    conv_dense_2 = Dense(10, activation='softmax')(conv_dropout_1)
    conv_opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    conv_model = Model(inputs=[input_image], outputs=[conv_dense_2])
    conv_model.compile(optimizer=conv_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    conv_model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_test, Y_test), shuffle=True)

