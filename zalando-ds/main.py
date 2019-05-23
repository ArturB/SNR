from __future__ import \
    absolute_import, \
    division, \
    print_function
import BatchImgDatasetFactory as imgf
import DatasetPlot as dsp
import tensorflow as tf

tf.enable_eager_execution()

if __name__ == '__main__':
    imgF = imgf.BatchImgDatasetFactory(
        batch_size_=32,
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
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "mobile.h5",
        period=1
    )
    history = model.fit(
        train_ds,
        steps_per_epoch=1000,
        validation_data=test_ds,
        validation_steps=150,
        callbacks=[checkpoint_callback],
        epochs=5
    )







