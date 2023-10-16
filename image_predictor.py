import numpy as np
import tensorflow as tf
import keras.utils as image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import sys
import time
import pixellib

from pixellib.tune_bg import alter_bg

reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/tensorflow_project/rps_nodropout.h5")

directory = "/home/pruthvi/Desktop/tensorflow_project/test_data/external_testing_data/"

for filename in os.listdir(directory): 
    f = os.path.join(directory, filename)
    print(f)

    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    change_bg.color_bg(f, colors = (0,128,0), output_image_name = "f_out.jpg")

# predicting images
    img = image.load_img("f_out.jpg", target_size=(150, 150))
    plt.imshow(img)
    plt.show()
    time.sleep(3)

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

        
