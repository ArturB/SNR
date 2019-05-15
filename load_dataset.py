from __future__ import \
    absolute_import, \
    division, \
    print_function
import keras
import pathlib
import random
import tensorflow as tf


tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_mobile_dataset(
        target_shape,
        batch_size=16,
        data_root_path="./dataset/_positive",
        file_format="jpg"
    ):
    """Load dataset from path, with given target shape and batch size."""
    data_root = pathlib.Path(data_root_path)
    all_image_paths = list(data_root.glob('*/*.' + file_format))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(lambda p: tf.image.decode_image(tf.read_file(p)), num_parallel_calls=AUTOTUNE)
    image_ds = image_ds.map(lambda i: tf.image.resize_image_with_pad(i, target_shape[0], target_shape[1]))

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    ds = image_label_ds.shuffle(buffer_size=batch_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def load_zalando_dataset(
    target_shape,
    batch_size = 16,
    ):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images







