#!/usr/bin/python3

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import sklearn

# Helper libraries

BATCH_SIZE = 16
TARGET_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 1)
VALIDATION_SPLIT = 0.3

def get_model_with_svm_layers():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    print(mobile_net2.summary())
    model = tf.keras.models.Sequential()
    model.add(mobile_net2)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(105, kernel_regularizer=tf.keras.regularizers.l2(0.01), name='predictions'))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    sk_model = sklearn.svm.SVC(kernel="linear", C=1e6, probability=True)

    return model


def test_sk_model():
    mobile_net2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights='imagenet')



def main(args):
    print('model preparation...')
    model = get_model_with_svm_layers()
    model.save('my_model.h5')
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
