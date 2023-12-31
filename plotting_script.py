import tensorflow as tf
import pandas as pd
from keras.callbacks import CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib as plt
import dataframe_image as dfi
from yellowbrick.classifier import ClassificationReport
from sklearn.neighbors import KNeighborsClassifier



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
    global validation_generator
    global y_pred

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

    #Making Classification Matrix

    print('\n\nClassification Report\n')
    target_names = ['Rock', 'Paper', 'Scissors']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    global crp
    crp = classification_report(validation_generator.classes, y_pred, target_names=target_names, output_dict=True)

evaluate(reconstructed_model)
print(cfm)
print(type(cfm))
print(cfm.shape)



#Making Confusion matrix look pretty
classes = ["Paper", "Rock", "Scissors"]

df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, cmap = "Purples", fmt=".3g")
plt.xlabel("Classifier Prediction")
plt.ylabel("Actual [Truth]")
cfm_plot.figure.savefig("confusion_matrix.png")

#Making Classification report look pretty

#Definition of terms:

#Precision - this is a measure of how exact the classifier is. (No.of true positives)/(Sum of true and false positives)
#"For all instances 'classified' positive, what percentage is correct?"


#Recall - this is a measure of how complete the classifier is : how able it is to correctly find all the true positive instances. (No. of true positives)/(Sum of true positives and false negatives)
#For all instances that are actually positive, what percentage was classified correctly?"


#F1 score - Harmonic mean of precision and recall: 2/(recall^(-1) + precision^(-1))

df_crp = pd.DataFrame(crp)
print(df_crp.to_string())
crp_plot = df_crp.iloc[:3, :3].T.plot(kind='bar',figsize=(12,7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
crp_plot.figure.savefig("classification_report.png")
plt.show()
