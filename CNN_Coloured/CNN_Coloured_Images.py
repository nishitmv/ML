import tensorflow as tf
from keras.optimizers.legacy.rmsprop import RMSProp
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import History
from keras.preprocessing.image import image_utils
import numpy as np

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

model.compile(loss= tf.losses.binary_crossentropy,
              optimizer=RMSProp(learning_rate=0.001),  # binary_crossentropy as only 2 types of images
              metrics=['accuracy'])  # root mean propagation (RMSprop), that takes a learning rate (lr)

history: History = model.fit(training_generator, epochs=10, validation_data=test_generator)

model.summary()

print(history.params)

img = image_utils.load_img(path="../horse-or-human/customData/HumanData/woman-1274056_640.jpg", target_size=(300, 300))
imgArray = image_utils.img_to_array(img)  # Convert Image to 2d Array
img3DArray = np.expand_dims(imgArray, axis=0)  # Add a third dimention to 2d array
imgVertical = np.vstack([img3DArray])  # stack the image vertically as training data is stacked vertically
predictions = model.predict(imgVertical)

print(predictions[0])

if (predictions[0] > 0.5):
    print("Image is a Human")
else:
    print("Image is a Horse")


img2 = image_utils.load_img(path="../horse-or-human/customData/HumanData/fashion-g134308119_1920.jpg", target_size=(300, 300))
imgArray2 = image_utils.img_to_array(img2)  # Convert Image to 2d Array
img3DArray2 = np.expand_dims(imgArray2, axis=0)  # Add a third dimention to 2d array
imgVertical2 = np.vstack([img3DArray2])  # stack the image vertically as training data is stacked vertically
predictions2 = model.predict(imgVertical2)

print(predictions2[0])

if (predictions2[0] > 0.5):
    print("Image is a Human")
else:
    print("Image is a Horse")

img3 = image_utils.load_img(path="../horse-or-human/customData/HorseData/horse-561221_640.jpg", target_size=(300, 300))
imgArray3 = image_utils.img_to_array(img3)  # Convert Image to 2d Array
img3DArray3 = np.expand_dims(imgArray3, axis=0)  # Add a third dimention to 2d array
imgVertical3 = np.vstack([img3DArray3])  # stack the image vertically as training data is stacked vertically
predictions3 = model.predict(imgVertical3)

print(predictions3[0])

if (predictions3[0] > 0.5):
    print("Image is a Horse")
else:
    print("Image is a Human")
