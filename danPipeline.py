# Implement a data Pipeline to test different input parameters

from getTrainingData import *
import time  # Allow time.time()
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import copy  # Using copy.deepcopy()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Architecture:
# Binarize
# Rescale
# PCA
# Multilayer Perceptron

# Also try feature selection with SelectKBest

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
            X[X>0] = 1.0
        return X

    def fit_transform(self,X,y=None,**params):
        #self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"binarize": self.binarize}

    def set_params(self, **parameters):
        self.binarize = parameters["binarize"]
        return self

def import_all_training_data(num_training_images=None):
    # Use read_csv to read training file into a dataframe
    labeled_images = pd.read_csv('./input/train.csv')

    # iloc selects data based on its integer position in the array
    # Alternatively, can use loc to select data by label type
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

    return X, y

# Get conditioned training data
def get_conditioned_training_data(num_training_images=None):
    X, y = import_all_training_data(num_training_images=num_training_images)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.9,random_state=0)
    return Xtrain, Xtest, ytrain, ytest

# Run basic pipeline
def run_basic_pipeline():
    Xtrain, Xtest, ytrain, ytest = get_conditioned_training_data()

    # Create classifiers
    bin = Binarizer()
    scaler = StandardScaler()
    pca = PCA(n_components=200)
    clf = MLPClassifier(hidden_layer_sizes=(100,100),alpha=0.0001)

    # Evaluate pipeline
    pip = Pipeline([('bin',bin), ('scaler', scaler), ('pca',pca), ('mlp', clf)])
    pip.fit(Xtrain,ytrain)

    print("Training accuracy:   ", "%.2f" % (100*pip.score(Xtrain,ytrain)), '%')
    print("Validation accuracy: ", "%.2f" % (100*pip.score(Xtest,ytest)), '%\n')

    return pip

# Run grid search over full pipeline
Xtrain, Xtest, ytrain, ytest = get_conditioned_training_data()

# Create classifiers
# bin = Binarizer()
# scaler = StandardScaler()
# pca = PCA()
# clf = MLPClassifier()
pipeline = Pipeline([
    ('binarizer',Binarizer()),
    ('scaler',StandardScaler()),
    ('pca',PCA()),
    ('mlp',MLPClassifier())
])

# Specify all parameters
param_set = {
    "binarizer__binarize" : [True,False],
    "scaler__with_mean": [True,False],
    "scaler__with_std": [True, False],
    "pca__whiten" : [True, False],
    "pca__n_components": [20,50,100,150,200,250,300],
    "mlp__alpha": [0.0,0.00001,0.0001,0.001],
    "mlp__hidden_layer_sizes": [(25,),(50,),(100,),(25,25),(50,50),(100,100),(25,25,25),(50,50,50),(100,100,100)]
}

# Create grid search
gs = GridSearchCV(pipeline, param_grid=param_set, refit=True)

# Run grid search
print("Begin training...")
t0 = time.time()
gs = gs.fit(Xtrain,ytrain)
tgrid = time.time() - t0
print("Total search time: ", "%.3f" % tgrid, "s")

# Get best classifier
print("Best params: ", gs.best_params_)
print("Best score:  ", gs.best_score_)




