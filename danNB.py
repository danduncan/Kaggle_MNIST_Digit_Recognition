# Examine MNIST dataset using various Naive Bayes models
# Training for NB is much faster than it was for SVM at large dataset sizes.
# Validation accuracy for 40000 images:
# Bernoulli NB    83.90%   Data were binarized
# Gaussian NB     56.44%   Data were not binarized
# Multinomial NB  83.24%   Data were not binarized

# Suppress warning for importing matplotlib
# import warnings;
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore");
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg

# Import other packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
import copy  # Using copy.deepcopy()
import time  # Using time.time()
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

# Print blank line to show script has started
print(' ')

# Print the learning setup
print("Model Used: ", NB_Models[modelIndex].__name__)
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

# Optional: View an image
# i=1
# img=train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.title(train_labels.iloc[i])
# plt.show()

# Show new histogram
# plt.hist(train_images.iloc[i])
# plt.show()

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


# OPTIONAL: Classify all test data and save to file ###

# Read in the test data and preprocess the same way the training data was processed
# test_data=pd.read_csv('./input/test.csv')
# test_data[test_data>0]=1
# results=clf.predict(test_data[0:5000])

# Output the results to a file
# df = pd.DataFrame(results)
# df.index.name='ImageId'
# df.index+=1
# df.columns=['Label']
# df.to_csv('results.csv', header=True)

# Signal process completed successfully
print("All done!")
