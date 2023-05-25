import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import image_utils, ImageDataGenerator
import numpy as np
import urllib.request
from keras.optimizers.legacy.rmsprop import RMSProp

training_data_gen = ImageDataGenerator(rescale=1 / 255)
training_dir = '../horse-or-human/training'
training_generator = training_data_gen.flow_from_directory(training_dir, target_size=(300, 300), class_mode='binary')

test_data_gen = ImageDataGenerator(rescale=1 / 255)
test_dir = '../horse-or-human/test'
test_generator = test_data_gen.flow_from_directory(test_dir, target_size=(300, 300), class_mode='binary')


weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(url=weights_url, filename=weights_file)

inceptionModel = InceptionV3(input_shape=(300, 300, 3), include_top=False, weights=None)
inceptionModel.load_weights(filepath=weights_file)

#  inceptionModel.summary()

for layer in inceptionModel.layers:
    layer.trainable = False

mixed7Layer = inceptionModel.get_layer('mixed7')
print("mixed7 layer shape : ", mixed7Layer.output_shape)
mixed7LayerOutput = mixed7Layer.output


flattenedMixed7Layer = tf.keras.layers.Flatten()(mixed7LayerOutput)
flattenedMixed7Layer = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(flattenedMixed7Layer)
flattenedMixed7Layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(flattenedMixed7Layer)

model = tf.keras.Model(inceptionModel.input, flattenedMixed7Layer)

model.compile(optimizer=RMSProp(learning_rate=0.001), loss='binary_crossentropy', metrics='accuracy')

history = model.fit(training_generator, epochs=2, validation_data=test_generator)

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