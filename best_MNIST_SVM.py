# Implement a Pipeline using a support vector classifier (SVC)
# Test accuracy should be 97.9%

import pandas as pd  # For pd.read_csv()
import time  # For time.time()
import copy  # For copy.deepcopy()
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib

class Binarizer(object):
    def __init__(self,binarize=True):
        # No internal variables
        self.binarize = binarize
        return

    def fit(self, X, y=None, **fit_params):
        # No need to fit
        return self

    def transform(self, X, **transform_params):
        if self.binarize == True:
            #X = copy.deepcopy(X)  # Needed when running in a BaggingClassifier
            X[X>0] = 1.0
            X = X.astype(float)  # This prevents warning being raised for float conversion
        return X

    def fit_transform(self,X,y=None,**params):
        #self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"binarize": self.binarize}

    def set_params(self, **parameters):
        self.binarize = parameters["binarize"]
        return self

def import_training_data(num_training_images=None,split=0.9):
    # Use read_csv to read training file into a dataframe
    labeled_images = pd.read_csv('./input/train.csv')

    # iloc selects data based on its integer position in the DataFrame
    X = None
    y = None
    if num_training_images is None:
        X = labeled_images.iloc[:, 1:]
        y = labeled_images.iloc[:, :1]
    else:
        X = labeled_images.iloc[0:num_training_images, 1:]
        y = labeled_images.iloc[0:num_training_images, :1]

    # Convert to matrices
    X = X.as_matrix()
    y = y.as_matrix().ravel()

    # Split into training and test data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=split, random_state=0)
    return Xtrain, ytrain, Xtest, ytest

# Create SVM pipeline
def get_svm_pipeline(Xtrain,ytrain,Xtest,ytest):
    # Best SVM:
    svm_pipeline = Pipeline([
        ('binarizer', Binarizer(binarize=True)),
        ('pca', PCA(n_components=50)),
        ('svc', SVC(kernel='rbf', decision_function_shape='ovo'))
    ])
    print("SVM pipeline created.")

    # Train the pipeline
    tstart = time.time()
    svm_pipeline.fit(Xtrain, ytrain)
    train_time = time.time() - tstart
    print("Train time: ", "%.3f" % train_time, "s")

    test_start_time = time.time()
    score = svm_pipeline.score(Xtest, ytest)
    test_time = time.time() - test_start_time
    print("Test time:  ", "%.3f" % test_time, "s")
    print("Test accuracy: ", "%.2f" % (100 * score), '%\n')

    return svm_pipeline

# Get input data
Xtrain, ytrain, Xtest, ytest = import_training_data(num_training_images=None)
training_set_size = len(ytrain)
print("Training set size: ", training_set_size, "images")

# Train pipeline components
svm_pipeline = get_svm_pipeline(Xtrain,ytrain,Xtest,ytest)

# Save trained SVM pipeline to file
#joblib.dump(svm_pipeline,"svm_pipeline.pkl")

# How to reload trained SVM pipeline from disk:
#svm_pipeline = joblib.load("svm_pipeline.pkl")
print("All done!")