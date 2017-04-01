# Examine MNIST dataset using Voting Classifier

from sklearn.ensemble import VotingClassifier
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
clf1 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_features=None)
clf2 = BernoulliNB()
clf3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2', C=1.0)
clf4 = PassiveAggressiveClassifier(loss='squared_hinge')
clf5 = Perceptron(penalty='l1', alpha=0.0001)
clf6 = MLPClassifier(hidden_layer_sizes=(100,100), alpha=0.0001)
clf7 = SVC(kernel='rbf')
estimators = [#("dt",clf1),
              #("bnb", clf2),
              ("lgr", clf3),
              #("pa", clf4),
              #("perc", clf5),
              ("mlp", clf6),
              ("svm", clf7) ]


# USER INPUTS #############################################

voting = 'hard' # hard or soft. Soft requires probabilistic classifiers
weights = None  # Can weigh the different classifiers if desired
n_jobs = -1     # Controls parallelism


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
print('Voting Classifier')
print('Number models: ', len(estimators))

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData, scaleData, pca_dimension)
print(' ')

tstart = time.time()

# Train the separate models
for i in range(len(estimators)):
    label, model = estimators[i]

    print("Model", i,": ", type(model).__name__)

    train_start_time = time.time()
    model.fit(train_images, train_labels.values.ravel())
    train_time = time.time() - train_start_time

    print("Train time: ", "%.3f" % train_time, "s")

    test_start_time = time.time()
    err = model.score(test_images, test_labels)
    test_time = time.time() - test_start_time

    print("Test time:  ", "%.3f" % test_time, "s")
    print("Test accuracy (with preprocessing): ", "%.2f" % (100*err), '%\n')

# Create voting model
vclf = VotingClassifier(estimators=estimators, voting=voting)

# Train voting model
train_start_time = time.time()
vclf.fit(train_images, train_labels.values.ravel())
train_time = time.time() - train_start_time

# Get the validation error
test_start_time = time.time()
err = vclf.score(test_images, test_labels)
test_time = time.time() - test_start_time

ttotal = time.time() - tstart

# Print runtime and accuracy
print("Voter train time: ", "%.3f" % train_time, "s")
print("Voter test time:  ", "%.3f" % test_time, "s")
print("Total time: ", "%.3f" % ttotal, "s")
print("Voter test accuracy: ", "%.2f" % (100*err), '%\n')

# Signal process completed successfully
print("All done!")
