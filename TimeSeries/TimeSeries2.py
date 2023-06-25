import keras.metrics
import numpy as np
import keras_tuner.engine.hyperparameters as hp
from keras_tuner import RandomSearch
from matplotlib import pyplot as plt
import tensorflow as tf


def plot_series(time, series, fmt="-", start=0, end=None):
    plt.plot(time[start: end], series[start: end], fmt)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def trend(time, slope=0.0):
    return time * slope


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype="float32")
series = trend(time, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

series = baseline + trend(time, slope)
series += seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset1 = tf.data.Dataset.from_tensor_slices(series)
    dataset1 = dataset1.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset1 = dataset1.flat_map(lambda window: window.batch(window_size + 1))
    dataset1 = dataset1.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset1 = dataset1.batch(batch_size).prefetch(1)
    return dataset1


split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=28, activation='relu', input_shape=[window_size]))
model.add(tf.keras.layers.Dense(28, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))

plot_series(time_train, x_train)


model.compile(loss="mse", optimizer="adam")

model.fit(dataset, epochs=50, verbose=1)
# tuner.search(dataset, epochs=100, verbose=0)
print(series[1000:1020])
print(model.predict(series[1000:1020][np.newaxis]))
print(series[1020])


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast



# forecastRes = model_forecast(model, series[..., np.newaxis], window_size)
# results = forecastRes[split_time - window_size:-1, -1, 0]
# print(results)