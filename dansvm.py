# Import matplotlib and suppress warnings
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg

# Import other packages
from sklearn import svm
from getTrainingData import *
import time  # Allow time.time()
# import sys # Allow exiting via sys.exit()
# import pandas as pd
# from sklearn.model_selection import train_test_split


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

# Print learning setup
print(' ')
print('Kernel: ', svmKernel)  # Inform user what kind of kernel is being used

# Read in training data
train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES, binarizeData)

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
