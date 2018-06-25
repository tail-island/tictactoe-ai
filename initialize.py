from funcy              import *
from keras.layers       import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from keras.models       import Model, save_model
from keras.regularizers import l2


def computational_graph():
    WIDTH  = 128
    HEIGHT =  16

    def ljuxt(*fs):
        return rcompose(juxt(*fs), list)

    def add():
        return Add()

    def batch_normalization():
        return BatchNormalization()

    def conv(channel_size):
        return Conv2D(channel_size, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

    def dense(unit_size):
        return Dense(unit_size, kernel_regularizer=l2(0.0005))

    def global_average_pooling():
        return GlobalAveragePooling2D()

    def relu():
        return Activation('relu')

    def softmax():
        return Activation('softmax')

    def tanh():
        return Activation('tanh')

    def residual_block():
        return rcompose(ljuxt(rcompose(batch_normalization(),
                                       conv(WIDTH),
                                       batch_normalization(),
                                       relu(),
                                       conv(WIDTH),
                                       batch_normalization()),
                              identity),
                        add())

    return rcompose(conv(WIDTH),
                    rcompose(*repeatedly(residual_block, HEIGHT)),
                    global_average_pooling(),
                    ljuxt(rcompose(dense(9), softmax()),
                          rcompose(dense(1), tanh())))


def main():
    model = Model(*juxt(identity, computational_graph())(Input(shape=(3, 3, 2))))
    model.summary()

    save_model(model, './model/0000.h5')
    save_model(model, './model/candidate/0000.h5')


if __name__ == '__main__':
    main()
