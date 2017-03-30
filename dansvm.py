# Import matplotlib and suppress warnings
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg

# Import other packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# import sys # Allow exiting via sys.exit()
import copy  # Allow deepcopy
import time  # Allow time.time()

# User inputs ############################################################

# Choose number of training images to use
# Runtime scales slightly faster than O(n^2)
NUM_TRAINING_IMAGES = 500

# Set the kernel function
# Results for N training images:
#         N=1000  N=10000 N=40000
# rbf     88.5%   91.85%  94.0%  <-- Runtime is several minutes for 40k examples
# linear  89.5%   90.1%
# poly    15.0%   54.0%
# sigmoid 83.5%   90.1%
svmKernel = 'rbf'

# Choose whether to binarize data
# SVM's perform significantly better when it is binarized
binarizeData = True

#######################################################################

# Print blank line to show script has started
print(' ')

# Images used are from MNIST dataset: 28x28 pixels
imageDim = 28

# Use read_csv to read training file into a dataframe
labeled_images = pd.read_csv('./input/train.csv')

# iloc selects data based on its integer position in the array
# Alternatively, can use loc to select data by label type
images = labeled_images.iloc[0:NUM_TRAINING_IMAGES, 1:]
labels = labeled_images.iloc[0:NUM_TRAINING_IMAGES, :1]

# Show how many images are in dataset and number used for training
print(labeled_images.size/(imageDim**2+1), 'total images in training set.')
print(images.size/(imageDim**2), 'images used for training.')

# Split training set into training and validation sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# Note that the split above does not copy the data,
# It merely creates a reference to the original. Need to manually run a deep copy:
train_images = copy.deepcopy(train_images)
test_images = copy.deepcopy(test_images)
train_labels = copy.deepcopy(train_labels)
test_labels = copy.deepcopy(test_labels)


# Clean up the data by rounding real values to either 0 or 1
if binarizeData:
    test_images[test_images > 0] = 1
    train_images[train_images > 0] = 1
    print('Data binarized')
else:
    print('Data not binarized')

print('Kernel: ', svmKernel)  # Inform user what kind of kernel is being used


# OPTIONAL: View an image (will need to uncomment matplotlib)
# i=1
# img=train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.title(train_labels.iloc[i])
# plt.show()

# Show new histogram
# plt.hist(train_images.iloc[i])
# plt.show()

# Train the SVM
clf = svm.SVC(kernel=svmKernel)
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
