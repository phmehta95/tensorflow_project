import tensorflow as tf
import classifier_script.py
from keras.callbacks import CSVLogger





#Plotting the Training and Testing accuracy                                    

reconstructed_model = tf.keras.models.load_model("/home/pruthvi/Desktop/tensorflow_project/rps.h5")
history = reconstructed_model.fit(classifier_script.train_generator, epochs=25, steps_per_epoch=20, validation_data = classifier_script.validation_generator, verbose = 1, validation_steps=3)



import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
