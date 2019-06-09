#!/usr/bin/python3

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.datasets import fashion_mnist
from keras import models, layers, optimizers
from copy import deepcopy

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
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

def get_model_with_dropped_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    out_layer = tf.keras.layers.ReLU(max_value=6., negative_slope=0.0, threshold=0.0)(mobile_net2.layers[-4].output)
    out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
    print(mobile_net2.layers[-1].get_config())
    out_layer = tf.keras.layers.Dense(105, activation='softmax', name='predictions')(out_layer)
    model = tf.keras.models.Model(mobile_net2.layers[0].input, out_layer)
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
    print(model.summary())
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
