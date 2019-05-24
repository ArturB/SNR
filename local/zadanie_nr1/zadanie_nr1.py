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


# Creates model with one trainable layer(last one)
def get_model_with_one_trainable_layer():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    print(mobile_net2.summary())
    for layer in mobile_net2.layers:
        layer.trainable = False
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(105, activation='softmax', name='predictions'))
    print(model.summary())
    return model

def main(args):
    print('model preparation...')
    model = get_model_with_one_trainable_layer()
    # model.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
