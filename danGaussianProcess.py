# Examine MNIST dataset using Logistic Regression and LogisticRegressionCV models

from sklearn.gaussian_process import GaussianProcessClassifier
from getTrainingData import *
import time  # Using time.time()
# import sys  # Allow use of sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split


# USER INPUTS #############################################

# Choose number of training images to use
NUM_TRAINING_IMAGES = 500

# Choose whether data gets binarized
binarizeData = True

##########################################################


# Print the learning setup
print(' ')
print('Model: Gaussian Process')

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData)

# Create model
clf = GaussianProcessClassifier(n_jobs=-1)  # n_jobs controls parallelism

# Train model
train_start_time = time.time()
clf.fit(train_images, train_labels.values.ravel())
train_time = time.time() - train_start_time

# Get the validation error
test_start_time = time.time()
err = clf.score(test_images, test_labels)
test_time = time.time() - test_start_time

# Print runtime and accuracy
print("Train time: ", "%.3f" % train_time, "s")
print("Test time:  ", "%.3f" % test_time, "s")
print("Test accuracy (with preprocessing): ", "%.2f" % (100*err), '%\n')

# Signal process completed successfully
print("All done!")
