#!/usr/bin/python3

from __future__ import \
    absolute_import, \
    division, \
    print_function
import math
import os
import random
import re
import subprocess
import tensorflow as tf
import urllib.request
import tarfile

BATCH_SIZE = 25
CPUs = tf.data.experimental.AUTOTUNE
EPOCHS = 10
IMG_SHAPE = (96, 96, 3)
JOB_NAME = "zadanie_nr1"
TRAIN_SET_SIZE = 128000
TEST_SET_SIZE = 14800
TRAIN_STEPS = math.floor(TRAIN_SET_SIZE / BATCH_SIZE)
TEST_STEPS = math.floor(TEST_SET_SIZE / BATCH_SIZE)


class BatchImgDatasetFactory:
    def __init__(self, batch_size_, image_shape_, output_range_):
        self.batch_size = batch_size_
        self.image_shape = image_shape_
        self.output_range = output_range_
        self.output_range_w = output_range_[1] - output_range_[0]

    def __make_batch_ds(self, image_ds_, labels_, init_transformer=lambda x: x, final_transformer=lambda x: x):
        image_ds = image_ds_
        image_ds = image_ds.map(init_transformer, num_parallel_calls=CPUs)
        image_ds = image_ds.map(
            lambda i: tf.image.resize_image_with_pad(i, self.image_shape[0], self.image_shape[1]),
            num_parallel_calls=CPUs
        )
        image_ds = image_ds.map(
            lambda x: (self.output_range_w * tf.cast(x, tf.float64) / 255.0) + self.output_range[0],
            num_parallel_calls=CPUs
        )
        image_ds = image_ds.map(final_transformer, num_parallel_calls=CPUs)

        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels_, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        ds = image_label_ds.shuffle(buffer_size=self.batch_size)
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.batch_size)

        return ds

    def from_dir(
            self,
            data_root_path,
            train_images_num,
            file_format=".jpg"
    ):
        def path_from_tensor(tensor):
            str_tens = str(tensor)
            str_tens = re.search("b'.*'", str_tens).group(0)
            str_tens = str_tens[2:-1]
            return str_tens

        print("Data root path: ", data_root_path)
        all_image_paths = tf.io.matching_files(data_root_path + "/*/*" + file_format)
        all_image_paths = [path_from_tensor(path) for path in all_image_paths]
        random.shuffle(all_image_paths)

        label_names = tf.io.matching_files(data_root_path + "/*")
        label_names = [path_from_tensor(path) for path in label_names]
        label_names = sorted(label_names)
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        print("Sample label: ", label_names[0])
        print("Sample key: ", os.path.dirname(all_image_paths[0]))
        all_image_labels = [label_to_index[os.path.dirname(path)] for path in all_image_paths]

        print("In ", data_root_path, " found ", len(all_image_paths), " images with ", len(label_names), " labels...")

        train_image_paths = all_image_paths[:train_images_num]
        train_image_labels = all_image_labels[:train_images_num]
        test_image_paths = all_image_paths[train_images_num:]
        test_image_labels = all_image_labels[train_images_num:]

        train_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
        train_ds = train_ds.map(
            lambda p: tf.image.decode_image(tf.read_file(p)),
            num_parallel_calls=CPUs
        )
        train_ds = self.__make_batch_ds(train_ds, train_image_labels)

        test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
        test_ds = test_ds.map(
            lambda p: tf.image.decode_image(tf.read_file(p)),
            num_parallel_calls=CPUs
        )
        test_ds = self.__make_batch_ds(test_ds, test_image_labels)

        return train_ds, test_ds, len(label_names)

    def zalando_dataset(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        def multiplicate_channel(t):
            t = tf.reshape([t, t, t], (3, 28, 28, 1))
            t = tf.transpose(t, [3, 1, 2, 0])
            t = tf.reshape(t, (28, 28, 3))
            return t

        train_ds = tf.data.Dataset.from_tensor_slices(train_images)
        train_ds = self.__make_batch_ds(train_ds, train_labels, init_transformer=multiplicate_channel)
        test_ds = tf.data.Dataset.from_tensor_slices(test_images)
        test_ds = self.__make_batch_ds(test_ds, test_labels, init_transformer=multiplicate_channel)

        return train_ds, test_ds, 10


tf.enable_eager_execution()

if __name__ == '__main__':
    print(subprocess.check_output("pwd"))
    urllib.request.urlretrieve("https://github.com/ArturB/SNR/releases/download/1.0.0/dataset.tar", "dataset-tar.tar")
    print("URL retrieve done!")
    td = tarfile.open("dataset-tar.tar")
    td.extractall()
    print("TAR extract done!")
    print(subprocess.check_output(["ls", "-l"]))

    imgF = BatchImgDatasetFactory(
        batch_size_=BATCH_SIZE,
        image_shape_=IMG_SHAPE,
        output_range_=(-1, 1)
    )
    train_ds, test_ds, label_num = \
        imgF.from_dir(
            data_root_path="dataset",
            train_images_num=TRAIN_SET_SIZE
        )

    ####################
    # MODEL DEFINITION #
    ####################

    mobile_net2 = \
        tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
    for layer in mobile_net2.layers:
        layer.trainable = False
    mobile_net2.layers[-2].trainable = True
    mobile_net2.layers[-3].trainable = True
    model = tf.keras.models.Sequential([
        mobile_net2,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(label_num, activation='softmax', name='predictions')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #############
    # END MODEL #
    #############

    board_callback = tf.keras.callbacks.TensorBoard(
        log_dir="gs://snr/" + JOB_NAME + "/logs",
        histogram_freq=1,
        batch_size=BATCH_SIZE,
        write_graph=True,
        write_grads=True,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='epoch'
    )
    history = model.fit(
        train_ds,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=test_ds,
        validation_steps=TEST_STEPS,
        verbose=1,
        callbacks=[board_callback],
        epochs=EPOCHS
    )

