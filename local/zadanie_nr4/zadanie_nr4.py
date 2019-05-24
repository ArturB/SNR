#!/usr/bin/python3

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.datasets import fashion_mnist
from keras import models, layers, optimizers

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 16
TARGET_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 1)
VALIDATION_SPLIT = 0.3

def get_model_with_all_trainable_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model

def get_model_with_dropped_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    mobile_net2.layers.pop(-4)
    mobile_net2.layers.pop(-5)
    print(mobile_net2.summary())
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model

def get_model_with_alpha(alpha=0.5):
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), alpha=alpha, include_top=False)
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model

def main(args):
    print('model preparation...')
    model = get_model_with_dropped_layers()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
