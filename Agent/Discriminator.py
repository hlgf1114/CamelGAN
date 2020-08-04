from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, Activation, Dropout
from keras.initializers import RandomNormal
from keras.models import Model

class Discriminator:

    def __init__(self, params):

        self.weight_init = RandomNormal(mean=0, stddev=0.02)

        self.discriminator_input = Input(shape=params.input_dim, name='discriminator_input')
        x = self.discriminator_input

        for i in range(4):

            x = Conv2D(
                filters=params.discriminator_conv_filters[i],
                kernel_size=params.discriminator_conv_kernel_size[i],
                strides=params.discriminator_conv_strides[i],
                padding='same',
                name='discriminator_conv_' + str(i)
            )(x)

            if params.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=params.discriminator_batch_norm_momentum)(x)

            x = Activation(params.discriminator_activation)(x)

            if params.discriminator_dropout_rate:
                x = Dropout(rate=params.discriminator_dropout_rate)(x)

        x = Flatten()(x)

        self.discriminator_output = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)

        self.discriminator = Model(self.discriminator_input, self.discriminator_output)


