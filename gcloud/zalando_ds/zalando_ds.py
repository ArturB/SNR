#!/usr/bin/python3

from __future__ import \
    absolute_import, \
    division, \
    print_function
from os import path
import random
import tensorflow as tf

CPUs = tf.data.experimental.AUTOTUNE


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
        data_root = path.Path(data_root_path)
        all_image_paths = list(data_root.glob('*/*' + file_format))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)

        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[path.Path(path).parent.name] for path in all_image_paths]

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


class PrintCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        if batch % 100 == 0:
            print('.')

    def on_epoch_end(self, epoch, logs):
        print(logs)


tf.enable_eager_execution()

if __name__ == '__main__':
    imgF = BatchImgDatasetFactory(
        batch_size_=25,
        image_shape_=(96, 96),
        output_range_=(-1, 1)
    )
    train_ds, test_ds, label_num = imgF.zalando_dataset()
    # dsp.plot_sample(train_ds, start_index=120)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False)
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(label_num, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    board_callback = tf.keras.callbacks.TensorBoard(
        log_dir="./logs",
        histogram_freq=0,
        batch_size=25,
        write_graph=True,
        write_grads=True,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='batch'
    )
    history = model.fit(
        train_ds,
        steps_per_epoch=2400,
        validation_data=test_ds,
        validation_steps=400,
        verbose=2,
        callbacks=[board_callback, PrintCallback()],
        epochs=10
    )


