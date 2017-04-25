from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, AveragePooling2D
from keras.optimizers import SGD
from keras import backend as K

import numpy as np
from PIL import Image, ImageOps
import argparse
import math
import os
import os.path
import glob

K.set_image_dim_ordering('th')  # ensure our dimension notation matches


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 8 * 8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 8, 8), input_shape=(128 * 8 * 8,)))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 128, 128)))  # Output features same size
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(4, 4)))  # W=128, F=4, S=4
    model.add(Convolution2D(128, 5, 5))  # W=32, F=5, S=1
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))  # W=28, F=2, S=2
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


model = generator_model()
print(model.summary())
model = discriminator_model()
print(model.summary())


def load_data(pixels=128, verbose=False):
    """
    Opens all the images in the logos folder, converts

    pixels: The resolution of the reshaped image, default = 128
    verbose: When True, displays the file path
    :return: A numpy array of all the training images, shape = (no. of images, 128, 128)
    """
    print("Loading data")
    X_train = []
    paths = glob.glob(os.path.normpath(os.getcwd() + '/logos/*.jpg'))
    for path in paths:
        if verbose:
            print(path)
        im = Image.open(path)
        im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
        im = ImageOps.grayscale(im)
        # im.show()
        im = np.asarray(im)
        X_train.append(im)
    print("Finished loading data")
    return np.array(X_train)


def train(epochs, batch_size, weights=False):
    """
    epochs: Train for this many epochs
    batch_size: Size of minibatch
    weights: If True, load weights from file, otherwise train the model from scratch.
    Use this if you have already saved state of the network and want to train it further.
    """
    X_train = load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Scale image to values from -1 to 1
    # introduce another dimension, not required for color images
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    if weights:
        generator.load_weights('goodgenerator.h5')
        discriminator.load_weights('gooddiscriminator.h5')
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((batch_size, 100))
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / batch_size))
        for index in range(int(X_train.shape[0] / batch_size)):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            generated_images = generator.predict(noise, verbose=0)
            # print(generated_images.shape)
            if index % 20 == 0 and epoch % 1 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                destpath = os.path.normpath(
                    os.getcwd() + "/logo-generated-images/" + str(epoch) + "_" + str(index) + ".png")
                Image.fromarray(image.astype(np.uint8)).save(destpath)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * batch_size)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if epoch % 10 == 9:
                generator.save_weights('goodgenerator.h5', True)
                discriminator.save_weights('gooddiscriminator.h5', True)


def generate_training_video(fps=1, path='./logo-generated-images'):
    """
    path: path to the generated images
    fps: frames per second for the video
    """
    os.system('ffmpeg -f image2 -r {} -i {}/%d0_0.png -vcodec mpeg4 -y movie.mp4'.format(fps, path))


def generate_video(frames=500, fps=10, path='./video_images'):
    """
    frames: Required number of frames, default = 500
    fps: Frames per second for the video, default = 10
    path: Path to store the generated video images, default = './video_images'

    Generates a video named dcgan.mp4 at the project root directory.
    """
    noise = np.random.uniform(-1, 1, (frames, 100))
    generator = generator_model()
    generator.load_weights('goodgenerator.h5')
    print("Generating images....")
    generated_images = generator.predict(noise, verbose=1)
    for i, image in enumerate(generated_images):
        image = image[0]
        image = image * 127.5 + 127.5
        clean(image)
        image = Image.fromarray(image.astype(np.uint8))
        image.save("video_images/video_images_{}.png".format(i))
    os.system('ffmpeg -f image2 -r {} -i {}/video_images_%d.png -vcodec mpeg4 -y dcgan.mp4'.format(fps, path))


def clean(image):
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i][j] + image[i + 1][j] + image[i][j + 1] + image[i - 1][j] + image[i][j - 1] > 127 * 5:
                image[i][j] = 255
    return image


def generate(batch_size):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('goodgenerator.h5')
    noise = np.zeros((batch_size, 100))
    for i in range(batch_size):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    # image = combine_images(generated_images)
    print(generated_images.shape)
    for image in generated_images:
        image = image[0]
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("dirty.png")
        Image.fromarray(image.astype(np.uint8)).show()
        clean(image)
        image = Image.fromarray(image.astype(np.uint8))
        image.show()
        image.save("clean.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

#train(400, 10, False)

generate(1)
