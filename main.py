from Agent.Network import Network
import os
import numpy as np
import matplotlib.pyplot as plt

epochs = 2000
batch_size = 64


myPath = os.path.join('./camel/data.npy')

x = np.load(myPath)

x = (x.astype('float32') - 127.5) / 127.5

x = x.reshape(x.shape[0], 28, 28, 1)

x_train = x[:8000]

def save_img(generator, epoch):

    noise = np.random.normal(0, 1, (16, 100))
    fake_img = generator.predict(noise)

    fake_list = []
    for i in fake_img:
        fake_list.append(i.reshape(28, 28))



    fig = plt.figure()
    for i in range(1, 17):
        ax = fig.add_subplot(4, 4, i)
        ax.imshow(fake_list[i - 1], cmap='gray')

    plt.savefig('./imgs/' + str(epoch) + '-img.png')


network = Network()

for epoch in range(epochs):

    d = network.train_discriminator(x_train, batch_size)
    g = network.train_generator(batch_size)

    print("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (
    epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

    if epoch % 100 == 0:
        save_img(network.generator.generator, epoch)
