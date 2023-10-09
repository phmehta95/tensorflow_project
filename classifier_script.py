#Classifier script
#Importing OS and python unzip
import os
import zipfile

#Unzipping training and testing data
local_zip = '/home/pruthvi/Desktop/tensorflow_project/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/pruthvi/Desktop/tensorflow_project/test_data/')
zip_ref.close()

local_zip = '/home/pruthvi/Desktop/tensorflow_project/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/pruthvi/Desktop/tensorflow_project/test_data/')
zip_ref.close()

#Checking number of files in training and testing dataset
rock_dir = os.path.join('test_data/rps/rock')
paper_dir = os.path.join('test_data/rps/paper')
scissors_dir = os.path.join('test_data/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

#Printing the names of the first 10 files in the training dataset
rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

#Iterate over the training images

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2


next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

print('ROCK NAME',next_rock)
print('PAPER NAME',next_paper)
print('SCISSORS NAME',next_scissors)
print(next_rock+next_paper+next_scissors)

#Plot the images
for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  print(i,img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.pause(1)
  plt.show(block=False) #block=False displays plot then goes to next line  
  plt.close()

#Importing tensorflow and the Keras NN model

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/home/pruthvi/Desktop/tensorflow_project/test_data/rps/"

#Keras Pre-processing: ImageDataGenerator is a Keras class which makes more images per image for training/testing by modifying them (shifting, scaling, resizing, rotating, zooming etc), below does this for the training and testing(validation) data sets

training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "/home/pruthvi/Desktop/tensorflow_project/test_data/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)



#Taking the data from the training and testing directory and setting the sizes of the images in no of pixels, and class_mode = binary if two classes (e.g. cats v dogs) or categorical if more than 2 classes, batch_size = n0. of images to be generated from the batch

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)


#Model training and building
from keras.callbacks import CSVLogger

 #Training the NN and building the model


#0) Type of model - Sequential. This allows us to build our model up layer by layer - this is just a linear stack of layers.

#1) Convolutional layers - core layer of the Convolutional NN. This takes an image as the input in its matrix representation (3 dimensional if its a colour image (RGB) and 2 dimensional if its a grayscale image). Dot product occurs between matrix and a filter (e.g. a Sobel filter) - filters look for features of the image - e.g. the Sobel filter looks for edges. Edges help with classification!

#2) Pooling layers - Neighbouring pixels tend to have similar values, so convolutional layers produce similar values for pixels next to each other in outputs. E.g. - if we have a filter that finds edges and it finds a strong edge at a certain location chances are that 1 pixel shifted over from this location, we'll find an edge too, but they're all the same edge! Pooling layers fix this problem by "pooling" all the values together in the output. The types of pooling are max, min and average. Max pooling puts the max value of a certain region into the output.

#3) Flatten layer - Turns the image height and width e.g. 5 x 5 into one dimension - 25 pixels long


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

csv_logger = CSVLogger("training.log", sep=",", append="False")

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3,callbacks=[csv_logger])

model.save("rps.h5")  #Saving to .h5 file (can be viewed in HDFview)









