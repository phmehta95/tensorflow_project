import tensorflow as tf
import pandas as pd
from keras.callbacks import CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib as plt


#Plotting the Training and Testing accuracy                                    

#Original CNN model
reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/tensorflow_project/rps.h5")
#CNN model with Dropout layer removed
#reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/tensorflow_project/rps_nodropout.h5")


#Training log for original model
log_data = pd.read_csv("training.log", sep=',' , engine='python')
#Training log with for no dropout model
#log_data = pd.read_csv("training2.log", sep=',' , engine='python')

#Plotting the Training and Testing accuracy
import matplotlib.pyplot as plt
acc = log_data['accuracy']
val_acc = log_data['val_accuracy']
loss = log_data['loss']
val_loss = log_data['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'm', label='Validation loss')


plt.title('Training and validation accuracy and loss')
plt.legend(loc=0)


plt.show()


#Function to plot a confusion matrix

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


TRAINING_DIR = "/home/pruthvi/Desktop/tensorflow_project/test_data/rps/"
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

cfm = 0

def evaluate(reconstructed_model):

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        class_mode='categorical',
        batch_size=126
     )

    
    validation_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
        class_mode='categorical',
        shuffle = False,
        batch_size=126)

    batch_size = 126
    num_test_samples = len(validation_generator.filenames)
    print (num_test_samples)
    Y_pred = reconstructed_model.predict_generator(validation_generator, num_test_samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    print(confusion_matrix(validation_generator.classes, y_pred))
    global cfm
    cfm = confusion_matrix(validation_generator.classes, y_pred)
    print(cfm)



evaluate(reconstructed_model)
print(cfm)
print(type(cfm))
print(cfm.shape)
#Making matrix look pretty



classes = ["Paper", "Rock", "Scissors"]

df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, cmap = "Purples", fmt=".3g")
plt.xlabel("Classifier Prediction")
plt.ylabel("Actual [Truth]")
cfm_plot.figure.savefig("confusion_matrix.png")

