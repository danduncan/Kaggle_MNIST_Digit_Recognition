# Examine MNIST dataset using collection of ExtraTrees

from sklearn.ensemble import ExtraTreesClassifier
from getTrainingData import *
import time  # Using time.time()
# import sys  # Allow use of sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split

criteria = ['gini','entropy'] # Criteria for determining quality of split
maxFeats = [None, 'sqrt', 'log2'] # Max number of features considered in each split

# USER INPUTS #############################################
critIndex = 1 # Criterion to use
featIndex = 2 # Use sqrt for random forest

n_estimators = 100 # Number of trees in forest
n_jobs = -1 # Sets parallelism to number of cores

# Choose number of training images to use
NUM_TRAINING_IMAGES = 40000

# Choose whether data gets binarized
binarizeData = True

##########################################################


# Print the learning setup
criterion = criteria[critIndex]
maxFeat = maxFeats[featIndex]
print(' ')
print('Model: Extra Trees Classifier')
print('Criterion: ', criterion)
print('Max Features: ', maxFeat)
print('N Estimators: ', n_estimators)


# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData)

# Create model
clf = ExtraTreesClassifier(criterion=criterion, max_features=maxFeat, n_estimators=n_estimators, n_jobs=n_jobs)

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
