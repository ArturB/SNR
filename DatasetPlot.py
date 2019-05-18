from __future__ import \
    absolute_import, \
    print_function
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_colorbar(ds, image_size, index=0):
    it = ds.make_one_shot_iterator()
    for i in range(index):
        t1, l1 = it.next()
    t1, l1 = it.next()
    t1 = tf.reshape(t1, (image_size[0], image_size[1], 3))

    plt.figure()
    plt.imshow(t1)
    plt.colorbar()
    plt.grid(False)

    plt.show()


def plot_sample(ds, image_size, start_index=0):
    it = ds.make_one_shot_iterator()
    for i in range(start_index):
        it.next()

    plt.figure(figsize=(10, 10))
    for i in range(25):
        t1, l1 = it.next()
        t1 = tf.reshape(t1, (t1.shape[1], t1.shape[2], 3))

        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(t1, cmap=plt.cm.binary)
        plt.xlabel(l1.numpy())
    plt.show()

