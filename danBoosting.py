# Examine MNIST dataset using Logistic Regression and LogisticRegressionCV models

from sklearn.ensemble import GradientBoostingClassifier
from getTrainingData import *
import time  # Using time.time()
# import sys  # Allow use of sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split

maxFeats = [None, 'sqrt', 'log2'] # Max number of features considered in each split

# USER INPUTS #############################################

featIndex = 1      # Max number of features to be used at each node
n_estimators = 100  # Default = 100
max_depth = 4       # Default = 3; requires tuning

# Choose number of training images to use
NUM_TRAINING_IMAGES = 500

# Choose whether data gets binarized
binarizeData = True

# Rescale data?
scaleData = True

# Dimensionality reduction via PCA?
# dim_reduction = None turns this off
# dim_reduction = up to number of pixels (784) turns this on and keeps that many dimensions
pca_dimension = None

##########################################################


# Print the learning setup
print(' ')
print('Model: Gradient Boosting')
print('Max Features: ', maxFeats[featIndex])
print('N Estimators: ', n_estimators)
print('Max Depth: ', max_depth)

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData, scaleData, pca_dimension)

# Create model
clf = GradientBoostingClassifier(max_features=maxFeats[featIndex], n_estimators=n_estimators, max_depth=max_depth)

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
