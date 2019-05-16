from __future__ import absolute_import, division, print_function
import dataset_loader as dl
import tensorflow as tf

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    ds, ts = dl.load_zalando_dataset((96, 96), batch_size=40)

    #dl.plot_one_image(ds, (96, 96), index=1002)
    #dl.plot_sample(ds, (96, 96), start_index=130)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False)
    mobile_net.trainable = False

    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

    model.fit(ds, epochs=100, steps_per_epoch=10)












