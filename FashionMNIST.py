import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

((training_images, training_labels), (test_images, test_labels)) = data.load_data()

print(training_images.shape)

training_images = training_images / 255
test_images = test_images / 255
print(training_images.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print("95% achieved")
            self.model.stop_training = True


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(training_images, training_labels, epochs=50, callbacks=AccuracyCallback())

model.evaluate(test_images, test_labels)
