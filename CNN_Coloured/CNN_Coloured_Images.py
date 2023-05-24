import tensorflow as tf
from keras.optimizers.legacy.rmsprop import RMSProp
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import History

training_data_gen = ImageDataGenerator(rescale=1 / 255)
training_dir = '../horse-or-human/training'
training_generator = training_data_gen.flow_from_directory(training_dir, target_size=(300, 300), class_mode='binary')

test_data_gen = ImageDataGenerator(rescale=1 / 255)
test_dir = '../horse-or-human/test'
test_generator = test_data_gen.flow_from_directory(test_dir, target_size=(300, 300), class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSProp(learning_rate=0.001),  # binary_crossentropy as only 2 types of images
              metrics=['accuracy'])  # root mean propagation (RMSprop), that takes a learning rate (lr)

history: History = model.fit(training_generator, epochs=10, validation_data=test_generator)

model.summary()

print(history.params)
