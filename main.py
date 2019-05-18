from __future__ import absolute_import, division, print_function
import BatchImgDatasetFactory as imgf
import DatasetPlot as dsp
import tensorflow as tf

tf.enable_eager_execution()

if __name__ == '__main__':
    imgF = imgf.BatchImgDatasetFactory(
        batch_size_=64,
        image_shape_=(96, 96),
        output_range_=(-1, 1)
    )
    train_zalando, test_zalando, label_num = imgF.zalando_dataset()
    # dsp.plot_sample(train_zalando, image_size=(96, 96), start_index=0)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(96, 96, 3)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(label_num, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_zalando, steps_per_epoch=300, epochs=5)












