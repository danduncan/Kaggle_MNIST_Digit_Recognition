# Examine MNIST dataset using AdaBoost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from getTrainingData import *
import time  # Using time.time()
# import sys  # Allow use of sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split


# Initialize relevant models
# Note: PasiveAggressive and MLP not supported by AdaBoost
models = [None,
          DecisionTreeClassifier(criterion='entropy', splitter='best', max_features=None),
          BernoulliNB(),
          LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2', C=1.0),
          Perceptron(penalty='l1', alpha=0.0001),
          SVC(kernel='rbf', probability=True)]


algorithms = ['SAMME.R', 'SAMME']

# USER INPUTS #############################################

modelIndex = 1     # Default is DecisionTree
n_estimators = 100  # Default is 50

algorithmIndex = 0  # Use 1 for Perceptron, 0 for all others


# Choose number of training images to use
NUM_TRAINING_IMAGES = 500

# Choose whether data gets binarized
binarizeData = True

# Rescale data?
scaleData = True

# Dimensionality reduction via PCA?
# dim_reduction = None turns this off
# dim_reduction = up to number of pixels (784) turns this on and keeps that many dimensions
pca_dimension = 200

##########################################################


# Print the learning setup
print(' ')
print("AdaBoost Classifier")
print('Model: ', type(models[modelIndex]).__name__)
print("Algorithm: ", algorithms[algorithmIndex])
print("N Estimators: ", n_estimators)


# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData, scaleData, pca_dimension)

# Create model
clf = AdaBoostClassifier(base_estimator=models[modelIndex], n_estimators=n_estimators, algorithm=algorithms[algorithmIndex])

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
