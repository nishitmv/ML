import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_series(time, series, fmt="-", start=0, end=None):
    plt.plot(time[start: end], series[start: end], fmt)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def getWeatherData():
    dataFile = 'UK_data.csv'
    fl = open(dataFile)
    data = fl.read()
    fl.close()
    lines = data.split("\n")
    temperatures = []
    for line in lines:
        if line:
            lineData = line.split(',')
            reading = lineData[1]
            if reading:
                temperatures.append(float(reading))

    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series


time, series = getWeatherData()

print(len(series))

mean = series.mean(axis=0)
series -= mean
std = series.std(axis=0)
series = series / std
start_time = 47450
split_time = 50370

time_training = time[start_time:split_time]
series_training = series[start_time:split_time]

time_validation = time[split_time + 1:]
series_validation = series[split_time + 1:]

plot_series(time_training, series_training)
plot_series(time_validation, series_validation)

window_size = 60
batch_size = 120
shuffle_buffer = 240
series_trn = tf.expand_dims(series_training, axis=-1)
dataset = tf.data.Dataset.from_tensor_slices(series_trn)
dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.batch(batch_size).prefetch(1)

valid_dataset = tf.data.Dataset.from_tensor_slices(series_validation)
valid_dataset = valid_dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
valid_dataset = valid_dataset.flat_map(lambda window: window.batch(window_size + 1))
valid_dataset = valid_dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
valid_dataset = valid_dataset.batch(batch_size).prefetch(1)

model = tf.keras.models.Sequential([
    #tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[None, 1]),
    #tf.keras.layers.SimpleRNN(100),
    tf.keras.layers.GRU(100, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.GRU(100),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.Huber(), optimizer="adam", metrics=["mae"])
model.fit(dataset, epochs=50, verbose=1, validation_data=valid_dataset)
