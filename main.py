from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import pathlib
import random
import tensorflow as tf

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
TARGET_SHAPE = (192, 192, 3)


def load_and_preprocess_image(path):
    return tf.read_file(path)


if __name__ == '__main__':
    data_root     = pathlib.Path("./dataset/_positive")
    data_root_neg = pathlib.Path("./dataset/_negative")
    all_image_paths = list(data_root.glob('*/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    print("Image count: ", image_count)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print("Labels (", len(label_names), "): ", label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Labels dictionary: ", label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    print("First 10 labels indices: ", all_image_labels[:10])

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(lambda p: tf.image.decode_image(tf.read_file(p)), num_parallel_calls=AUTOTUNE)
    image_ds = image_ds.map(lambda i: tf.image.resize_image_with_pad(i, TARGET_SHAPE[0], TARGET_SHAPE[1]))

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # image_label_ds = image_label_ds.filter(lambda t, _: tf.equal(t.shape, (200, 200, 3)))

    ds = image_label_ds.shuffle(buffer_size=BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    print(ds)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
    mobile_net.trainable = False
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(label_names))])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])

    # keras_ds = ds.map(lambda t, l: (2*t-1,l))

    model.fit(ds, epochs=1, steps_per_epoch=1000)












