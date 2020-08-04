from Agent.Generator import Generator
from Agent.Discriminator import Discriminator
from Agent.Params import Params
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model

import numpy as np

class Network:

    def __init__(self):

        self.params = Params(
            (28,28,1),
            [64,64,128,128],
            [5,5,5,5],
            [2,2,2,1],
            None,
            'relu',
            0.4,
            0.0008,
            (7,7,64),
            [2,2,1,1],
            [128,64,64,1],
            [5,5,5,5],
            [1,1,1,1],
            0.9,
            'relu',
            None,
            0.0004,
            'rmsprop',
            100
        )

        self.discriminator = Discriminator(self.params)
        self.generator = Generator(self.params)

        # 판별자 컴파일
        self.discriminator.discriminator.compile(
            optimizer=RMSprop(lr=0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 생성자 컴파일
        self.discriminator.discriminator.trainable = False
        model_input = Input(shape=(self.params.z_dim,), name='model_input')
        model_output = self.discriminator.discriminator(self.generator.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer=RMSprop(lr=0.0004),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.discriminator.discriminator.trainable = True

    def train_discriminator(self, x_train, batch_size):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # 진짜 이미지로 훈련
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

        d_loss_real, d_acc_real = self.discriminator.discriminator.train_on_batch(true_imgs, valid)

        # 생성된 이미지로 훈련
        noise = np.random.normal(0, 1, (batch_size, self.params.z_dim))
        gen_imgs = self.generator.generator.predict(noise)

        d_loss_fake, d_acc_fake = self.discriminator.discriminator.train_on_batch(gen_imgs, fake)

        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)


        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):

        vaild = np.ones((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, self.params.z_dim))
        return self.model.train_on_batch(noise, vaild)