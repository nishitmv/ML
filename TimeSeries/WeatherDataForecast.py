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
    dataFile = 'station.csv'
    fl = open(dataFile)
    data = fl.read()
    fl.close()
    lines = data.split("\n")
    headers = lines[0].split(",")
    lines = lines[1:]
    temperatures = []
    for line in lines:
        if line:
            lineData = line.split(',')
            lineData = lineData[1:13]
            for reading in lineData:
                if reading:
                    temperatures.append(float(reading))
    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series


time, series = getWeatherData()

plot_series(time, series)

print(len(time))

mean = series.mean(axis=0)
series -= mean
std = series.std(axis=0)
series = series / std

split_time = 1680

time_training = time[:split_time]
series_training = series[:split_time]

time_validation = time[split_time + 1:]
series_validation = series[split_time + 1:]

plot_series(time_training, series_training)
plot_series(time_validation, series_validation)

window_size = 24
batch_size = 12
shuffle_buffer = 48
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

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=28, activation='relu', input_shape=[window_size]))
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                                 strides=1, padding="causal", activation="relu",
                                 input_shape=[None, 1]))
model.add(tf.keras.layers.Dense(28, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))

model.compile(loss="mse", optimizer="adam")

model.fit(dataset, epochs=50, verbose=1, validation_data=valid_dataset)

# forecastRes = model_forecast(model, time_validation[..., np.newaxis], window_size)
# results = forecastRes[split_time - window_size:-1, -1, 0]
# print(forecastRes)
