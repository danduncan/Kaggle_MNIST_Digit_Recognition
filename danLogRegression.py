# Examine MNIST dataset using Logistic Regression and LogisticRegressionCV models

# Import other packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
import copy  # Using copy.deepcopy()
import time  # Using time.time()
# import sys  # Allow use of sys.exit()

# Possible models
models = [LogisticRegression, LogisticRegressionCV]

# List the different options for descent algorithms
# liblinear: Best for small datasets. Only solver that supports L1 regularization,
#            but does not natively support multinomial classification
# ncg and lbfgs support large datasets
# sag best for very large datasets
solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag']

# USER INPUTS #############################################

# Choose desired model and solver by their indexes (lists are zero-indexed)
modelIndex = 1
solverIndex = 2

# Regularization Parameters:
regPenalty = 'l2' # Can be l1 or l2 (only liblinear supports l1)
regParameter = 1.0 # Float from 0 to infinity. Smaller value = stronger regularization. Default is 1.0
                   # Not used by LogisticRegressionCV

# Choose to use one-versus-rest or multinomial classification.
# Multinomial is not supported by liblinear solver
multi = 'multinomial' # Options: 'ovr' or 'multinomial'

# Choose number of training images to use
NUM_TRAINING_IMAGES = 40000

# Choose whether data gets binarized
binarizeData = True

##########################################################


# START SCRIPT ###

# Print blank line to show script has started
print(' ')

# Print the learning setup
clfName = models[modelIndex].__name__
print("Model Used: ", clfName)
print("Solver Used: ", solvers[solverIndex])
print("Classification: ", multi)
print('Regularization: ', regPenalty, regParameter)
if binarizeData:
    print("Data binarized.")
else:
    print("Data not binarized.")

# Images used are from MNIST dataset: 28x28 pixels
imageDim = 28

# Use read_csv to read training file into a dataframe
labeled_images = pd.read_csv('./input/train.csv')

# Show how many images are in dataset
# Each image is 785 elements: 1 label, then 28*28 = 784 pixels
print(labeled_images.size/(imageDim**2+1), 'total images in training set.')

# iloc selects data based on its integer position in the array
# Alternatively, can use loc to select data by label type
images = labeled_images.iloc[0:NUM_TRAINING_IMAGES, 1:]
labels = labeled_images.iloc[0:NUM_TRAINING_IMAGES, :1]

# Output how many images are actually being used
print(images.size/(imageDim**2), 'images used for training.')

# Split training set into training and validation sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# Not that the split above does not copy the data
# It merely creates a reference to the original. Need to manually deep copy:
train_images = copy.deepcopy(train_images)
test_images = copy.deepcopy(test_images)
train_labels = copy.deepcopy(train_labels)
test_labels = copy.deepcopy(test_labels)

# Clean up the data by rounding real values to either 0 or 1
if binarizeData:
    test_images[test_images > 0] = 1
    train_images[train_images > 0] = 1


# Train model
clf = None
if clfName == 'LogisticRegression':
    clf = LogisticRegression(solver=solvers[solverIndex], penalty=regPenalty, C=regParameter, multi_class=multi)
else:
    clf = LogisticRegressionCV(solver=solvers[solverIndex], penalty=regPenalty, multi_class=multi)

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
