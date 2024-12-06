import os
import numpy as np
import pandas as pd
from matplotlib.image import imread
import matplotlib.pyplot as plt

#NEURALNET TRAINING CODE

#VARIABLES
NUM_OF_IMGS = 1600   #specify number of images of each character
IMG_DIMS = 32       #specify h x w dimensions of image
BATCH_SIZE = 16     #specify batch size
EPOCHS = 500


#DATA IMPORTATION AND MANIPULATION
#Letters I want
letterList = ["character_1_ka", "character_2_kha", "character_3_ga", "character_4_gha",
              "character_5_kna", "character_6_cha", "character_7_chha", "character_8_ja",
              "character_9_jha", "character_10_yna", "character_11_taamatar", "character_12_thaa"]

#Append these letters with the training path
trainList = []
for i in letterList:
    trainList.append("Train/"+i)

#Append these letters with the Test path
testList = []
for i in letterList:
    testList.append("Test/"+i)


# Declare a numpy array with the amount I want, and the image size.
data = np.ndarray(shape=(len(trainList)*NUM_OF_IMGS, IMG_DIMS, IMG_DIMS), dtype=np.float32)
count = 0
for i in trainList: #For every path we made (Ex: "Train/character_1_ka")
    count2 = 0
    for x in os.listdir(i): #Using the os.listdir function, I am able to get every image inside each folder
        data[count2+(NUM_OF_IMGS*count)] = imread(i+"/"+x) #Use the imread function from matplotlib to read the image. Example input is i = "Train/character_1_ka" and x = "10962.png"
        count2+=1
        if(count2 == NUM_OF_IMGS): #This is so we only pull 120 images per letter folder.
            break #This puts us from character_1_ka to character_2_kha
    count+=1 #Counting how long our array is

#reshape data to a 2-d array (converting images from 2-d to 1-d)
data = data.reshape(len(data), -1)

#insert labels into column 0
#labeling them numerically as they are defined "character_1" and so on
label = 0
data_new = []
for row in data:
    modified_row = np.insert(row, 0, int(label/NUM_OF_IMGS))
    data_new.append(modified_row)
    label += 1

data = np.array(data_new)

#get shape of array
m, n = data.shape

#shuffle before splitting in to dev and training sets
np.random.shuffle(data)

#split into train/dev/test (80/10/10)
#transpose data for input into NN
total_sample = len(data)
train = data[:int(total_sample*0.8)].T
dev = data[int(total_sample*0.8):int(total_sample*0.9)].T
test = data[int(total_sample*0.9):].T

#SETUP DEVELOPMENT DATA TO BE INPUT TO NEURALNET
#get labels
Y_dev = np.array(dev[0]).astype(int)
#get pixel values
X_dev = dev[1:n]
#standardize pixel values to a value between 0-1
X_dev = X_dev

#SETUP TRAINING DATA TO BE INPUT TO NEURALNET
#get labels
Y_train = np.array(train[0]).astype(int)
#get pixel values
X_train = train[1:n]
#standardize pixel values to a value between 0-1
X_train = X_train
#get Y_train max value for one_hot_encoding
Y_MAX = Y_train.max()

#SETUP TESTING DATA TO BE INPUT TO NEURALNET
#get labels
Y_test = np.array(test[0]).astype(int)
X_test = test[1:n]
#standardize pixel values to a value between 0-1
X_test = X_test

#'''
###################################################
#DEFINE FUNCTIONS FOR NN TRAINING
###################################################

#Randomly generate initial weights and baises
def init_params():
    W1 = np.random.rand(len(trainList), IMG_DIMS*IMG_DIMS) - 0.5
    b1 = np.random.rand(len(trainList), 1) - 0.5
    W2 = np.random.rand(len(trainList), len(trainList)) - 0.5
    b2 = np.random.rand(len(trainList), 1) - 0.5
    return W1, b1, W2, b2

print(init_params())

#activation function
def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

#probability distribution for prediction
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

#forward propogation of weights and baises
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#encode categorical variables as numerical
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y_MAX + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#take results of forward_prop and pass back through NN
#to adjust weights and baises
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

#update weights and biases after backward_prop
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

#make predictions based off results of softmax function in forward_prop
def get_predictions(A2):
    return np.argmax(A2, 0)

#compare predictions with known labels and return ratio to determine results
def get_accuracy(predictions, Y):
    print("Predicitons:\n",predictions, "\nLabels:\n", Y)
    return np.sum(predictions == Y) / Y.size

#perform NN training
def gradient_descent(X, Y, alpha):
    W1, b1, W2, b2 = init_params()
    #loop for epochs
    for i in range(EPOCHS):
        #loop for batchs
        for j in range(int(len(X[0])/BATCH_SIZE)):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X[:,BATCH_SIZE*j:BATCH_SIZE*(j+1)])
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X[:,BATCH_SIZE*j:BATCH_SIZE*(j+1)], Y[BATCH_SIZE*j:BATCH_SIZE*(j+1)])
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            #display current accuracy every 10 epochs
        if i % 10 == 0:
            print("Epoch: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ",get_accuracy(predictions, Y[-BATCH_SIZE:]))
    
    #BLOCK TO DISPLAY LOSS & VAL_LOSS
    '''

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(24,12))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    '''

    return W1, b1, W2, b2

#CALLS
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1)
#'''