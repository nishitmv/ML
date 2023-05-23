import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

print(training_images.shape)

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),  # rectified linear unit ensures
    # filtering outputs and picks only >0. (3, 3) is a 3 x 3 filter , 64 = number of convolutions ( filters ) .
    tf.keras.layers.MaxPooling2D(2, 2),  # Size of max pool matrix( max pool picks max value from n x n matrix)
    tf.keras.layers.Flatten(),  # Flatten 2d array to 1d array
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Number of Neurons
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Max value for 1 of ten items this code is trying to find
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(training_images, training_labels, epochs=20)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

