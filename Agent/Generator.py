from keras.layers import Input, Dense, UpSampling2D, BatchNormalization, Activation, Reshape, Dropout, Conv2D, Conv2DTranspose
from keras.initializers import RandomNormal
from keras.models import Model
import numpy as np
class Generator:

    def __init__(self, params):

        self.weight_init = RandomNormal(mean=0, stddev=0.02)

        self.generator_input = Input(shape=(params.z_dim,), name='generator_input')
        x = self.generator_input

        x = Dense(np.prod(params.generator_initial_dense_layer_size))(x)

        if params.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=params.generator_batch_norm_momentum)(x)
        x = Activation(params.generator_activation)(x)

        x = Reshape(params.generator_initial_dense_layer_size)(x)

        if params.generator_dropout_rate:
            x = Dropout(rate=params.generator_dropout_rate)(x)

        for i in range(4):

            if params.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=params.generator_conv_filters[i],
                    kernel_size=params.generator_conv_kernel_size[i],
                    strides=params.generator_conv_strides[i],
                    padding='same',
                    name='generator_conv_' + str(i),
                )(x)
            else:
                x = Conv2DTranspose(
                    filters=params.generator_conv_filters[i],
                    kernel_size=params.generator_conv_kernel_size[i],
                    padding='same',
                    strides=params.generator_conv_strides[i],
                    name='generator_conv_' + str(i),
                    kernel_initializer=self.weight_init
                )(x)

            if i < 4 - 1:
                if params.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum=params.generator_batch_norm_momentum)(x)
                x = Activation('relu')(x)
            else:
                x = Activation('tanh')(x)

        self.generator_output = x
        self.generator = Model(self.generator_input, self.generator_output)

