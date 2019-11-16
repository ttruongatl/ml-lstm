from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

print('tensorflow version: {}'.format(tf.__version__))


def __prepare_train_data(df, feature):
    groups = df.groupby(['event', 'start'])
    data = []
    labels = []
    for id, group in groups:
        values = group['CylinderBorePressure'].values
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(values, (len(values), 1)))
        labels.append(id[0])
    return np.array(data), np.array(convert_labels(labels))


def convert_labels(labels):
    digit_labels = []
    for label in labels:
        if label == 'cut':
            digit_labels.append(0.0)
        elif label == 'sort':
            digit_labels.append(1.0)
        elif label == 'idle':
            digit_labels.append(2.0)

    return digit_labels


TRAIN_SPLIT = 300000
BATCH_SIZE = 256
BUFFER_SIZE = 10000

tf.random.set_seed(13)

train_df = pd.read_csv('data/st-cloud.csv')
train_df = train_df.sort_values(by=['timestamp'])
train_df = train_df.loc[(train_df['event'] == 'cut') | (train_df['event'] == 'sort') | (train_df['event'] == 'idle')]
x_train_uni, y_train_uni = __prepare_train_data(train_df, feature='CylinderBorePressure')

print(x_train_uni[0])
print(y_train_uni[0])

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
# train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
# val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
# val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
#
# simple_lstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
#     tf.keras.layers.Dense(1)
# ])
#
# simple_lstm_model.compile(optimizer='adam', loss='mae')
#
# for x, y in val_univariate.take(1):
#     print(simple_lstm_model.predict(x).shape)
#
# EVALUATION_INTERVAL = 200
# EPOCHS = 10
#
# simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
#                       steps_per_epoch=EVALUATION_INTERVAL,
#                       validation_data=val_univariate, validation_steps=50)

# for x, y in val_univariate.take(3):
#   plot = show_plot([x[0].numpy(), y[0].numpy(),
#                     simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
#   plot.show()
