import keras.metrics
import numpy as np
from matplotlib import pyplot as plt


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
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

valid = series[0: 461]
valid_time = time[0: 461]
naive_forecast = series[1000 - 1:-1]

forecast_time = time[1000 - 1: -1]
plot_series(naive_forecast, forecast_time)
plot_series(valid, valid_time)

print(keras.metrics.mean_squared_error(valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(valid, naive_forecast).numpy())


