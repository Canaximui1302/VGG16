import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pathlib 
import tensorflow as tf
import numpy as np
import datetime



data_dir = pathlib.Path("./flower_photos")
print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count: " + str(image_count))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"] )
print("Class names: ") 
print(CLASS_NAMES)

output_class_units = len(CLASS_NAMES)
print("Output units: " + str(output_class_units))



def block1(inp):
    inp = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(inp)
    return inp

def block2(inp):
    inp = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(inp)
    return inp

def block3(inp):
    inp = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.MaxPooling2D((2, 2), strides  = (2, 2))(inp)
    return inp

def block4(inp):
    inp = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(inp)
    return inp

def block5(inp):
    inp = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', padding = "same")(inp)
    inp = tf.keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(inp)
    return inp



BATCH_SIZE = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227
Steps_per_epoch = int(np.floor(image_count/BATCH_SIZE))

img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = img_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))



def vg16(shape, batch_size = BATCH_SIZE):
    input = tf.keras.Input(shape, batch_size)

    inp = block1(input)
    inp = block2(inp)
    inp = block3(inp)
    inp = block4(inp)
    inp = block5(inp)

    # Classification block
    inp = tf.keras.layers.Flatten(name='flatten')(inp)
    inp = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(inp)
    inp = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(inp)
    inp = tf.keras.layers.Dense(output_class_units, activation='softmax', name='predictions')(inp)

    model = tf.keras.Model(input, inp)
    return model



model = vg16((227,227,3), BATCH_SIZE)

model.compile(optimizer = 'sgd', loss = "categorical_crossentropy", metrics = ['accuracy'])

model.summary()



# Training the Model
history = model.fit(
      train_data_gen,
      steps_per_epoch=Steps_per_epoch,
      epochs=20)

# Saving the model
model.save('VGG16_saved_model.keras')
