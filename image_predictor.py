import numpy as np
import tensorflow as tf
import keras.utils as image

reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/tensorflow_project/rps.h5")

# predicting images
path = "/home/pruthvi/Downloads/rockhand.jpg"
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = reconstructed_model.predict(images, batch_size=10)
print(classes)

if classes.flat[0] == 1.0:
    print("PAPER")

if classes.flat[1] == 1.0:
    print("ROCK")

if classes.flat[2] == 1.0:
    print("SCISSORS")

        
