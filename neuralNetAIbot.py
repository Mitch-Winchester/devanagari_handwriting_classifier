from neuralNetTraining import X_test, Y_test, letterList, IMG_DIMS
from neuralNetTraining import forward_prop, get_predictions
from matplotlib import pyplot as plt

#make predictions with dataset passed into method
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

#use test set to test results of training
#print predictions, actual labels, and image
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_test[:, index, None]
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
    label = Y_test[index]
    print("Prediction: ", letterList[int(prediction)])
    print("Label: ", letterList[int(label)])
    current_image = current_image.reshape((IMG_DIMS, IMG_DIMS)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
