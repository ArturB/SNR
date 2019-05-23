#!/usr/bin/python3

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 16
TARGET_SIZE = (224, 224)
VALIDATION_SPLIT = 0.3

def get_model_with_two_trainable_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in mobile_net2.layers:
        layer.trainable = False
    mobile_net2.layers[-2].trainable = True
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    # print(model.summary())
    return model


def get_model_with_three_trainable_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in mobile_net2.layers:
        layer.trainable = False
    mobile_net2.layers[-2].trainable = True
    mobile_net2.layers[-3].trainable = True
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    # print(model.summary())
    return model


def get_model_with_four_trainable_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in mobile_net2.layers:
        layer.trainable = False
    mobile_net2.layers[-2].trainable = True
    mobile_net2.layers[-3].trainable = True
    mobile_net2.layers[-4].trainable = True
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    # print(model.summary())
    return model


def get_model_with_five_trainable_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in mobile_net2.layers:
        layer.trainable = False
    mobile_net2.layers[-2].trainable = True
    mobile_net2.layers[-3].trainable = True
    mobile_net2.layers[-4].trainable = True
    mobile_net2.layers[-5].trainable = True
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model


def main(args):
    print('model preparation...')
    model = get_model_with_five_trainable_layers()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
