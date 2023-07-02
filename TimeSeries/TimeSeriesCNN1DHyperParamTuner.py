import keras.metrics
import numpy as np
from keras_tuner import RandomSearch
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras_tuner


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
    series = tf.expand_dims(series, axis=-1)
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
hp = keras_tuner.HyperParameters()

model = keras.Sequential()
# model.add(tf.keras.layers.Dense(units=28, activation='relu', input_shape=[window_size]))
# filterParam = hp.Int('units', min_value=128, max_value=256, step=64)
# kernel_size_param = hp.Int('kernels', min_value=3, max_value=9, step=3)
# strides_param = hp.Int('strides', min_value=1, max_value=3, step=1)
model.add(keras.layers.Conv1D(filters=hp.Int(name='units', min_value=128, max_value=256, step=64),
                              kernel_size=hp.Int(name='kernels', min_value=3, max_value=9, step=3),
                              strides=hp.Int(name='strides', min_value=1, max_value=3, step=1), padding="causal",
                              activation="relu",
                              input_shape=[None, 1]))
model.add(keras.layers.Dense(28, input_shape=[window_size], activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.relu))
model.add(keras.layers.Dense(1))
model.compile(loss="mse", optimizer="adam", metrics="loss")
# plot_series(time_train, x_train)




# model.fit(dataset, epochs=50, verbose=1)
# tuner.search(dataset, epochs=100, verbose=0)
# print(series[1000:1020])
# print(model.predict(series[1000:1020][np.newaxis]))
# print(series[1020])

tuner = RandomSearch(hypermodel=model, objective='loss',
                     max_trials=500, executions_per_trial=3,
                     directory='my_dir', project_name='cnn-tune')
tuner.search_space_summary()
tuner.search(dataset, epochs=100, verbose=2)

# forecastRes = model_forecast(model, series[..., np.newaxis], window_size)
# results = forecastRes[split_time - window_size:-1, -1, 0]
# print(results)
# plot_series(time, forecastRes)
