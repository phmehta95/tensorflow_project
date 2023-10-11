import tensorflow as tf
import pandas as pd
from keras.callbacks import CSVLogger





#Plotting the Training and Testing accuracy                                    

reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/tensorflow_project/rps.h5")
#history = reconstructed_model.fit(classifier_script.train_generator, epochs=25, steps_per_epoch=20, validation_data = classifier_script.validation_generator, verbose = 1, validation_steps=3)

log_data = pd.read_csv("training.log", sep=',' , engine='python')


import matplotlib.pyplot as plt
acc = log_data['accuracy']
val_acc = log_data['val_accuracy']
loss = log_data['loss']
val_loss = log_data['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
