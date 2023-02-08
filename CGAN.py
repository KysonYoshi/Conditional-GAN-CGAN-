from __future__ import print_function, division

import cv2
import os
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

size=300

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = size
        self.img_cols = size
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2
        self.latent_dim = 100

        optimizer = Adam(0.0005, 0.5)
        optimizer2 = Adam(0.0001, 0.5)


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer2,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=32, sample_interval=50):

        path = './confirmed_fronts'
        X_train = []
        y_train = []
        c=0
        """for folder in os.listdir(path):
            for img_name in os.listdir(path + '/' + folder+'/2017'):
                img_path = path + '/' + folder + '/2017' +'/'+ img_name
                print(img_path)
                img = cv2.imread(img_path)
                img_resize = cv2.resize(img, (200, 200))
                X_train.append(img_resize)
                y_train.append(c)
            c+=1
            if c>=6:
                break"""
        for img_name in os.listdir('./confirmed_fronts/Audi/2017'):
            img_path = './confirmed_fronts/Audi/2017'+'/'+ img_name
            print(img_path)
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, (size, size))
            X_train.append(img_resize)
            y_train.append(0)
        for img_name in os.listdir('./confirmed_fronts/BMW/2018'):
            img_path = './confirmed_fronts/BMW/2018'+'/'+ img_name
            print(img_path)
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, (size, size))
            X_train.append(img_resize)
            y_train.append(1)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # Load the dataset

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch%100==0:
                self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 1, 2
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0,2).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        img_resize = cv2.resize(gen_imgs[0, :, :], (size, size))
        #img_resize=gen_imgs[0, :, :]
        im = Image.fromarray((img_resize * 255).astype(np.uint8))
        im.save("./output/Audi/%d.jpg" % (epoch))
        img_resize = cv2.resize(gen_imgs[1, :, :], (size, size))
        # img_resize=gen_imgs[0, :, :]
        im = Image.fromarray((img_resize * 255).astype(np.uint8))
        im.save("./output/BMW/%d.jpg" % (epoch ))


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=7001, batch_size=32, sample_interval=200)