#!/usr/bin/python3

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.datasets import fashion_mnist
from keras import models, layers, optimizers

from image_preparation import image_preparation
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 16
TARGET_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 1)
VALIDATION_SPLIT = 0.3

def basic_model_preparation():
    print('model preparation...')
    vgg16_conv = VGG16(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    print(vgg16_conv.summary())
    model = models.Sequential()
    model.add(vgg16_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dense(4096, activation='relu', name='fc2'))
    model.add(layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model

def main(args):

    print('image preparation...')
    train_generator = image_preparation(BATCH_SIZE, TARGET_SIZE, VALIDATION_SPLIT, color_mode="grayscale")
    model = basic_model_preparation()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

    print('compile...')
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    print('fit...')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=10,
        validation_steps=153378*VALIDATION_SPLIT/train_generator.batch_size,
        verbose=1)

    print('results...')
    print(history.history['acc'])
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
