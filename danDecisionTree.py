# Examine MNIST dataset using Passive Aggressive Classifier

from sklearn.tree import *
from getTrainingData import *
import time  # Using time.time()
# import sys  # Allow use of sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split

models = [DecisionTreeClassifier, ExtraTreeClassifier] # Extra tree is a hyper-random decision tree
criteria = ['gini','entropy'] # Criteria for determining quality of split
splitters = ['best','random'] # Criteria to choose the split at each node
maxFeats = [None, 'sqrt', 'log2'] # Max number of features considered in each split

# USER INPUTS #############################################
modelIndex = 1 # For ExtraTree, always use splitter=random and maxFeats=sqrt
critIndex = 1 # Criterion to use
splitIndex = 1 # Splitter to choose splits
featIndex = 1 # Choose max number of features at each split

# Choose number of training images to use
NUM_TRAINING_IMAGES = 40000

# Choose whether data gets binarized
binarizeData = True

##########################################################


# Print the learning setup
model = models[modelIndex]
criterion = criteria[critIndex]
splitter = splitters[splitIndex]
maxFeat = maxFeats[featIndex]
print(' ')
print('Model: ', model.__name__)
print('Criterion: ', criterion)
print('Splitter: ', splitter)
print('Max Featurs: ', maxFeat)


# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData)

# Create model
clf = model(criterion=criterion, splitter=splitter, max_features=maxFeat)

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
