import numpy as np
from keras.layers import Dense
from keras import Sequential

l0 = Dense(1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
xy = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, xy, epochs=500)

print(model.predict([10.0]))
print(l0.get_weights())
