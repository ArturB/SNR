#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from keras.applications.vgg19 import VGG19
from keras.datasets import fashion_mnist
from keras import models, layers, optimizers

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    vgg19_conv = VGG19()
    print(vgg19_conv.summary())

    model = models.Sequential()

    for layer in vgg19_conv.layers[:-2]:
        model.add(layer)

    for layer in model.layers[:-2]:
        layer.trainable = False

    model.add(layers.Dense(10, activation='softmax'))

    print(model.summary())


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # model.fit(train_images, train_labels, epochs=5)
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    # print('Test accuracy:', test_acc)
    # predictions = model.predict(test_images)
    # for layer in model.layers:
    #     print(layer, layer.trainable)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
