# Examine MNIST dataset using various Naive Bayes models
# Training for NB is much faster than it was for SVM at large dataset sizes.
# Validation accuracy for 40000 images:
# Bernoulli NB    83.90%   Data were binarized
# Gaussian NB     56.44%   Data were not binarized
# Multinomial NB  83.24%   Data were not binarized

from sklearn.naive_bayes import *
import time  # Using time.time()
from getTrainingData import *
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import copy  # Using copy.deepcopy()
# import sys  # Allow use of sys.exit()


# List the different options for Naive Bayes
NB_Models = [BernoulliNB, GaussianNB, MultinomialNB]

# USER INPUTS #############################################

# Choose desired model by its index (lists are zero-indexed)
modelIndex = 2

# Choose number of training images to use
NUM_TRAINING_IMAGES = 500

# Choose whether data gets binarized
binarizeData = False

##########################################################

# START SCRIPT ###

# Print the learning setup
print(' ')
print("Model Used: ", NB_Models[modelIndex].__name__)

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData)

# Train Gaussia Naive Bayes on the cleaned up data
clfName = NB_Models[modelIndex]
clf = clfName()
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
