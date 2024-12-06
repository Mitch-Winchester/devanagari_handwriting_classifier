from neuralNetTraining import W1, b1, W2, b2, X_dev, Y_dev, X_test, get_accuracy
from neuralNetAIbot import test_prediction, make_predictions
import random

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("\nAccuracy on dev_set: ",get_accuracy(dev_predictions, Y_dev))

#loop to check test images against training data
num_of_imgs = 10
for i in range(num_of_imgs):
    rand_img = random.randint(0,len(X_test[0]))
    test_prediction(rand_img, W1, b1, W2, b2)