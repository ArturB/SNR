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
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    print("Image count: ", image_count)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print("Labels (", len(label_names), "): ", label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Labels dictionary: ", label_to_index)




