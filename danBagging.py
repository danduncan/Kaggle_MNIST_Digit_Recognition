# Examine MNIST dataset using Logistic Regression and LogisticRegressionCV models

from sklearn.ensemble import BaggingClassifier
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
models = [DecisionTreeClassifier(criterion='entropy', splitter='best', max_features=None),
          BernoulliNB(),
          LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2', C=1.0),
          PassiveAggressiveClassifier(loss='squared_hinge'),
          Perceptron(penalty='l1', alpha=0.0001),
          MLPClassifier(hidden_layer_sizes=(100,100), alpha=0.0001),
          SVC(kernel='rbf')]


# USER INPUTS #############################################

modelIndex = 6     # Default is DecisionTree
n_estimators = 5  # Default is 10
n_jobs = -1        # Controls parallelism


# Choose number of training images to use
NUM_TRAINING_IMAGES = 40000

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
print('Model: ', type(models[modelIndex]).__name__)
print("N Estimators: ", n_estimators)

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData, scaleData, pca_dimension)

# Create model
clf = BaggingClassifier(base_estimator=models[modelIndex], n_estimators=n_estimators, n_jobs=n_jobs)

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
