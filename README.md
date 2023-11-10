# tensorflow_project

Hi! This is a little project I've been working on which uses a CNN (Convolutional Neural Net) to classify images of hands making gestures of either rock, paper, or scissors!

Training dataset: https://storage.googleapis.com/learning-datasets/rps.zip Testing dataset: https://storage.googleapis.com/learning-datasets/rps-test-set.zip

It contains the following scripts:

A) classifier_script.py -> This is the main script which loads in the testing and training datasets and trains the model. It uses Tensorflow (an open-source platform for machine learning and a symbolic math library that is used for machine learning applications) and Keras (an Open Source Neural Network library which has a very user friendly API) :)

The model layers are explained as comments in the code, but I will also list them here:
0) Type of model - Sequential. This allows us to build our model up layer by layer - this is just a linear stack of layers.
1) Convolutional layers - core layer of the Convolutional NN. This takes an image as the input in its matrix representation (3 dimensional if its a colour image (RGB) and 2 dimensional if its a grayscale image). Dot product occurs between matrix and a filter (e.g. a Sobel filter) - filters look for features of the image - e.g. the Sobel filter looks for edges. Edges help with classification!
2) Pooling layers - Neighbouring pixels tend to have similar values, so convolutional layers produce similar values for pixels next to each other in outputs. E.g. - if we have a filter that finds edges and it finds a strong edge at a certain location chances are that 1 pixel shifted over from this location, we'll find an edge too, but they're all the same edge! Pooling layers fix this problem by "pooling" all the values together in the output. The types of pooling are max, min and average. Max pooling puts the max value of a certain region into the output.
3) Flatten layer - Turns the image height and width e.g. 5 x 5 into one dimension - 25 pixels long.
The trained model is saved to a HDF5 (.h5) file, so it can be used in different scripts and applications without having to wait the long training times every time you want to use or test the model. An example model (rps.h5) has been uploaded.




B) plotting_script.py -> This script is responsible for producing plots which allow the user to visualise the performance of their model. This is done by way of the:
1) Training and validation accuracy and loss plot - Perfomance (accuracy and loss) of the model when training and testing (validation) is shown plotted against the number of epochs (an epoch is one cycle of the model through the training dataset). Also uploaded are these plots made with and without the Dropout layer in the CNN - you can see that without the Dropout layer the validation loss becomes greater than the validation accuracy, which is a symptom of overfitting! This means that when it come to validating the model using the testing dataset, the CNN has focussed on a particular feature of the training dataset so much that it cannot identify unseen data very well. A Dropout layer in the CNN prevents this because the Dropout layer means that the probability of certain inputs not being used in an epoch is <1 (e.g. Dropout(0.5) means that 50% of the time the probability of certain random inputs is set to 0) so that overtraining is unlikely to happen!

2) Classification matrix - This is a nice handy visualisation of how well the classifier works on the testing dataset, with actual (truth) info on whether the testing image was rock/paper/scissors on the y-axis and the prediction on the x-axis.
3) Classification report - This gives a table and an additional bar chart on metrics used to test how well the classifier works, namely, precision, recall and f1-score. These concepts are explained in the code.




C) image_predictor.py -> This script takes external testing data (e.g. a user's uploaded image of their hand) to classify it. PixelLib is used to give the background a block colour to help the classifier focus on the hand in the image. The prediction is output as a simple print statement. 
