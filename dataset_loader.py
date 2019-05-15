from __future__ import \
    absolute_import, \
    print_function
import keras
import pathlib
import random
import tensorflow as tf


def load_dataset(
        data_root_path,
        train_images_num,
        target_shape,
        batch_size=16,
        file_format=".jpg"
):
    data_root = pathlib.Path(data_root_path)
    all_image_paths = list(data_root.glob('*/*' + file_format))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    train_image_paths = all_image_paths[:train_images_num]
    train_image_labels = all_image_labels[:train_images_num]
    test_image_paths = all_image_paths[train_images_num:]
    test_image_labels = all_image_labels[train_images_num:]

    train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_image_ds = train_path_ds.map(
        lambda p: tf.image.decode_image(tf.read_file(p)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_image_ds = train_image_ds.map(
        lambda i: tf.image.resize_image_with_pad(i, target_shape[0], target_shape[1])
    )
    train_image_ds = train_image_ds.map(lambda x: tf.divide(tf.cast(x, tf.float16), 255.0))

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
    test_image_ds = test_path_ds.map(
        lambda p: tf.image.decode_image(tf.read_file(p)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_image_ds = test_image_ds.map(
        lambda i: tf.image.resize_image_with_pad(i, target_shape[0], target_shape[1])
    )
    test_image_ds = test_image_ds.map(lambda x: tf.divide(tf.cast(x, tf.float16), 255.0))

    train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_image_labels, tf.int64))
    test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_image_labels, tf.int64))

    train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
    test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))

    train_ds = train_image_label_ds.shuffle(buffer_size=batch_size)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_ds = test_image_label_ds.shuffle(buffer_size=batch_size)
    test_ds = test_ds.repeat()
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds


def load_zalando_dataset(
    target_shape,
    batch_size=16,
):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_image_ds = tf.data.Dataset.from_tensor_slices(train_images)
    test_image_ds = tf.data.Dataset.from_tensor_slices(test_images)

    train_image_ds = train_image_ds.map(lambda x: tf.reshape([x, x, x], (28, 28, 3)))
    train_image_ds = train_image_ds.map(lambda x: tf.divide(tf.cast(x, tf.float16), 255.0))
    train_image_ds = train_image_ds.map(
        lambda i: tf.image.resize_image_with_pad(i, target_shape[0], target_shape[1])
    )
    test_image_ds = test_image_ds.map(lambda x: tf.reshape([x, x, x], (28, 28, 3)))
    test_image_ds = test_image_ds.map(lambda x: tf.divide(tf.cast(x, tf.float16), 255.0))
    test_image_ds = test_image_ds.map(
        lambda i: tf.image.resize_image_with_pad(i, target_shape[0], target_shape[1])
    )

    train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
    test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))

    train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
    test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))

    train_ds = train_image_label_ds.shuffle(buffer_size=batch_size)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_ds = test_image_label_ds.shuffle(buffer_size=batch_size)
    test_ds = test_ds.repeat()
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds
