from __future__ import absolute_import, division, print_function
import dataset_loader as dl
import tensorflow as tf

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    ds, ts = dl.load_zalando_dataset((96, 96))

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False)
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(105)])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

    model.fit(ds, epochs=10, steps_per_epoch=8000)












