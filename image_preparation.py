#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import pathlib
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# tf.enable_eager_execution()
# AUTOTUNE = tf.data.experimental.AUTOTUNE



def load_and_preprocess_image(path):
    return tf.read_file(path)

def image_preparation(batch_size, target_size, validation_split=0.1, color_mode='rgb'):
    train_data_dir = "./common-mobile-web-app-icons"
    train_datagen = ImageDataGenerator(validation_split=validation_split)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                          target_size=target_size,
                                          color_mode=color_mode,
                                          classes=None,
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=True,
                                          seed=None,
                                          interpolation='nearest')
    return train_generator


