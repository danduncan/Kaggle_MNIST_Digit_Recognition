# Examine MNIST dataset using Logistic Regression and LogisticRegressionCV models

from sklearn.linear_model import *
from getTrainingData import *
import time  # Using time.time()
# import sys  # Allow use of sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split

# Possible models:
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
NUM_TRAINING_IMAGES = 500

# Choose whether data gets binarized
binarizeData = True

##########################################################


# Print the learning setup
print(' ')
clfName = models[modelIndex].__name__
print("Model Used: ", clfName)
print("Solver Used: ", solvers[solverIndex])
print("Classification: ", multi)
print('Regularization: ', regPenalty, regParameter)

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData)

# Create model
clf = None
if clfName == 'LogisticRegression':
    clf = LogisticRegression(solver=solvers[solverIndex], penalty=regPenalty, C=regParameter, multi_class=multi)
else:
    clf = LogisticRegressionCV(solver=solvers[solverIndex], penalty=regPenalty, multi_class=multi)

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
