class Params:

    def __init__(
            self,
            input_dim,
            discriminator_conv_filters,
            discriminator_conv_kernel_size,
            discriminator_conv_strides,
            discriminator_batch_norm_momentum,
            discriminator_activation,
            discriminator_dropout_rate,
            discriminator_learning_rate,
            generator_initial_dense_layer_size,
            generator_upsample,
            generator_conv_filters,
            generator_conv_kernel_size,
            generator_conv_strides,
            generator_batch_norm_momentum,
            generator_activation,
            generator_dropout_rate,
            generator_learning_rate,
            optimizer,
            z_dim
                 ):
        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        self.optimizer = optimizer
        self.z_dim = z_dim

