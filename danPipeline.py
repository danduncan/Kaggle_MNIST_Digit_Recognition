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
def get_conditioned_training_data(num_training_images=None,split=True):
    Xtrain, ytrain = import_all_training_data(num_training_images=num_training_images)

    Xtest = None
    ytest = None

    if split == True:
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xtrain, ytrain, train_size=0.9,random_state=0)

    return Xtrain, ytrain, Xtest, ytest

# Get the N best models from the GridSearch sorted from best to worst
def get_gridsearch_top_models(gs=None, numModels=10):

    # Get total number of models trained
    totalModels = len(gs.cv_results_['mean_train_score'])

    # If total models is less than numModels, reduce numModels
    numModels = min(totalModels,numModels)

    # Get detailed results
    cv = gs.cv_results_
    params_list = cv['params']
    params_rank = cv['rank_test_score']
    params_test_score = cv['mean_test_score']
    params_train_score = cv['mean_train_score']
    params_test_std = cv['std_test_score']
    params_score_time = cv['mean_score_time']

    # Create list of statistics of top models
    topModels = {
        'params': (),
        'index': np.empty(numModels),
        'train_score': np.empty(numModels),
        'test_score': np.empty(numModels),
        'test_std': np.empty(numModels),
        'score_time': np.empty(numModels)
    }

    # Populate results
    count = 0  # count is required to deal with ties
    rankIndex = 1
    while count < numModels:
        # Get an array of indexes for the current rank
        # Note that if there is a tie, there may end up being an empty array or one with multiple indices
        indexes = np.where(params_rank == rankIndex)[0]
        for i in range(len(indexes)):
            if count >= numModels:
                break
            currentIndex = indexes[i]
            topModels['index'][count] = currentIndex
            topModels['params'] += (params_list[currentIndex],)
            topModels['train_score'][count] = params_train_score[currentIndex]
            topModels['test_score'][count] = params_test_score[currentIndex]
            topModels['test_std'][count] = params_test_std[currentIndex]
            topModels['score_time'][count] = params_score_time[currentIndex]
            count += 1
        rankIndex += 1

    # Condense parameters list
    params = topModels['params']
    keys = list(topModels['params'][0].keys())

    # Create dictionary to store condensed parameters
    params_condensed = {}
    for i in range(len(keys)):
        params_condensed[keys[i]] = []

    # Build up condensed list for all parameters in order of descending model rank
    for i in range(numModels):
        for j in range(len(keys)):
            params_condensed[keys[j]].append(params[i][keys[j]])

    topModels['params_condensed'] = params_condensed


    return topModels

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
Xtrain, ytrain, Xtest, ytest = get_conditioned_training_data(split=False)


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
    "binarizer__binarize" : [True],
    "scaler__with_mean": [True,False],
    "scaler__with_std": [False],
    "pca__whiten" : [False],
    "pca__n_components": [20,50,100,150,200], #,250,300],
    "mlp__alpha": [0.0,0.00001,0.0001,0.001],
    "mlp__hidden_layer_sizes": [(100,),(200,),(100,100),(200,200),(100,100,100),(200,200,200)]
}

# Create grid search
gs = GridSearchCV(pipeline, param_grid=param_set, refit=True, n_jobs=4, pre_dispatch='2*n_jobs', verbose=0)

# Run grid search
print("Begin training...")
t0 = time.time()
gs = gs.fit(Xtrain,ytrain)
tgrid = time.time() - t0
print("\nTotal search time: ", "%.3f" % tgrid, "s")
print("Total models tested: ", "%.0f" % len(gs.cv_results_['mean_train_score']))

# Get best classifier
print("\nBest params: ", gs.best_params_)
print("Best validation score:  ", "%.4f" % gs.best_score_)

# Extract top models
topModels = get_gridsearch_top_models(gs,numModels=20)
print("Top validation scores: \n", topModels['test_score'])

# Run classifier on test data
t0 = time.time()
score = gs.score(Xtest, ytest)
ttest = time.time() - t0
print("\nBest classifier test time: ", "%.2f" % ttest, "s")
print("Best classifier test accuracy: ", "%.2f" % (100*score), "%")


# Best classifier so far:
# Binarize = True
# scaler: mean=False, std=False
# PCA = 50, whiten=False
# MLP alpha=0.001, layers=(100,)
# Validation accuracy = 97.24%