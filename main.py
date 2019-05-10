from __future__ import absolute_import, division, print_function
import IPython.display as display
import pathlib
import random

import tensorflow as tf
tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    data_root     = pathlib.Path("./dataset/_positive")
    data_root_neg = pathlib.Path("./dataset/_negative")
    all_image_paths = list(data_root.glob('*/*.JPG'))
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

    # img_path = all_image_paths[0]
    # print(img_path)
    # img_raw = tf.read_file(img_path)
    # img_tensor = tf.image.decode_image(img_raw)
    # print(img_tensor.shape)
    # print(img_tensor.dtype)

    all_image_tensors = [tf.image.decode_image(tf.read_file(img_path)) for img_path in all_image_paths[:10000]]
    all_200x200_tensors = list(filter(lambda x: x.shape == (200,200,3), all_image_tensors))
    print(len(all_200x200_tensors))





