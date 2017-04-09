# Implement a data Pipeline to test different input parameters

from getTrainingData import *
import time  # Allow time.time()
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import *
import os
from sklearn.metrics import classification_report
import copy  # Using copy.deepcopy()


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

    # Include total number of models that were analyzed
    topModels['total_models_tested'] = totalModels

    return topModels


# Convert params to string
# Params is a tuple of dicts
def params_to_string(params):
    numModels = len(params)

    str = ""

    for i in range(numModels):
        p = params[i]
        str += "Model " + i.__str__() + ":\n"
        keys = p.keys()
        for key in keys:
            str += "\t" + key + ":\t" + p[key].__str__() + "\n"

        str += "\n"

    return str

# Given pipeline gridsearch, get string of class names for steps in pipeline
def pipeline_gridsearch_to_string(gs):
    pip = gs.estimator
    step = pip.steps
    str = "Pipeline Architecture:\n"

    for i in range(len(step)):
        str += "\t" + (i+1).__str__() +".  " + step[i][1].__class__.__name__ + "\n"

    str += "\n"
    return str

# Save topModels to file with other statistics
def save_stats_to_file(gs,topModels,stats,model=None):
    # Lines 1-4:
    # Maximum accuracy
    # Training set size
    # Total models trained
    # Total train time

    if model is None:
        model = 'score'

    # Filename should be based on best score
    score_rounded = int(10000*stats['best_score'])
    fileroot0 = "results/gridsearch_" + model + "_" + score_rounded.__str__()
    fileroot = fileroot0

    filename = fileroot + ".txt"
    i = 1
    while os.path.isfile(filename) == True:
        # Do not overwrite existing files
        fileroot = fileroot0 + "_" + i.__str__()
        filename = fileroot + ".txt"
        i+=1


    f = open(filename,'w+')

    # Write first 4 lines
    f.write("Max test accuracy: %.3f" % (100*stats['best_score']) + "% \n")
    f.write("Training set size %.0f" % stats['training_set_size'] + "\n")
    f.write('Total models trained: %.0f' % stats['number_models'] + "\n")
    f.write('Train time: %.2f' % stats['train_time'] + "s\n\n")

    # Write pipeline architecture
    f.write(pipeline_gridsearch_to_string(gs))

    # Write more detailed info
    f.write("Best model parameters: \n" + topModels['params'][0].__str__() + "\n\n")
    f.write("All Train Scores:\n" + topModels['train_score'].__str__() + "\n\n")
    f.write("All Test Scores:\n" + topModels['test_score'].__str__() + "\n\n")
    f.write("All Test Std Devs:\n" + topModels['test_std'].__str__() + "\n\n")
    f.write("All Score Times:\n" + topModels['score_time'].__str__() + "\n\n")
    f.write("All model parameters:\n" + params_to_string(topModels['params']))

    # Close file
    f.close()

    # Save the gridsearch object to file too
    # Object can be loaded back into memory by:
    #   gs = joblib.load("filename.pkl")
    joblib.dump(gs, fileroot + ".pkl")

    return f

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
def pipeline_gridsearch(classifier,param_set=None,num_training_images=None):
    Xtrain, ytrain, Xtest, ytest = get_conditioned_training_data(num_training_images=num_training_images,split=False)
    training_set_size = len(ytrain)
    print("Training set size: ", training_set_size, "images")

    # Create grid search
    gs = GridSearchCV(classifier, param_grid=param_set, refit=True, n_jobs=6, pre_dispatch='2*n_jobs', verbose=0)

    # Run grid search
    print("Begin training...")
    t0 = time.time()
    gs = gs.fit(Xtrain, ytrain)
    trainTime = time.time() - t0
    print("\nTotal search time: ", "%.3f" % trainTime, "s")
    print("Total models tested: ", "%.0f" % len(gs.cv_results_['mean_train_score']))

    # Get best classifier
    print("\nBest params: ", gs.best_params_)
    print("Best validation score:  ", "%.4f" % gs.best_score_)

    # Compile important statistics
    stats = {
        'train_time': trainTime,
        'training_set_size': training_set_size,
        'best_score': gs.best_score_,
        'number_models': len(gs.cv_results_['mean_train_score'])
    }

    # Return the gridsearch object
    return gs, stats

def mlp_pipeline_gridsearch():
    # Create classifiers
    pipeline = Pipeline([
        ('binarizer', Binarizer()),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('mlp', MLPClassifier())
    ])

    # Specify all parameters
    param_set = {
        "binarizer__binarize": [True],
        "scaler__with_mean": [False],
        "scaler__with_std": [False],
        "pca__whiten": [False],
        "pca__n_components": [50, 100],  # ,20,150,200,250,300],
        "mlp__alpha": [0.0, 0.00001, 0.0001, 0.001],
        "mlp__hidden_layer_sizes": [(500,), (500, 500), (500, 500, 500), (500, 500, 500, 500)]
    }

    # Run gridsearch
    gs, stats = pipeline_gridsearch(pipeline, param_set=param_set, num_training_images=None)

    # Extract top models
    topModels = get_gridsearch_top_models(gs, numModels=20)
    print("Top validation scores: \n", topModels['test_score'])

    # Save stats to file
    f = save_stats_to_file(gs, topModels, stats, model="MLP")

    return gs, stats, topModels

# Run a gridsearch with an SVM pipeline
def svm_pipeline_gridsearch():
    # Create classifiers
    pipeline = Pipeline([
        ('binarizer',Binarizer()),
        ('scaler',StandardScaler()),
        ('pca',PCA()),
        ('svc',SVC())
    ])

    # Specify all parameters
    param_set = {
        "binarizer__binarize" : [True],
        "scaler__with_mean": [False],
        "scaler__with_std": [False],
        "pca__whiten" : [False],
        "pca__n_components": [20,50,100,200], #150,200,250,300],
        "svc__kernel": ["rbf"],
        "svc__decision_function_shape": ['ovr']
    }

    print("Pipeline with SVM")

    # Run gridsearch
    gs, stats = pipeline_gridsearch(pipeline,param_set=param_set,num_training_images=None)

    # Extract top models
    topModels = get_gridsearch_top_models(gs,numModels=20)
    print("Top validation scores: \n", topModels['test_score'])

    # Save stats to file
    f = save_stats_to_file(gs,topModels,stats,model="SVM")

    return gs, stats, topModels

# Run a gridsearch on random forests
def random_forest_pipeline_gridsearch():
    # Create classifiers
    pipeline = Pipeline([
        ('binarizer',Binarizer()),
        ('scaler',StandardScaler()),
        ('pca',PCA()),
        ('rf',RandomForestClassifier())
    ])

    # Specify all parameters
    param_set = {
        "binarizer__binarize" : [True],
        "scaler__with_mean": [True,False],
        "scaler__with_std": [True,False],
        "pca__whiten" : [True,False],
        "pca__n_components": [20,50], # ,100,200], #150,200,250,300],
        "rf__criterion": ["gini","entropy"],
        "rf__max_features": ["sqrt","log2"], # [None,"sqrt","log2"]
        "rf__n_estimators": [40,50,60], # [5,10,20,30,40,50],
        "rf__n_jobs": [1]
    }

    print("Pipeline with RandomForest")

    # Run gridsearch
    gs, stats = pipeline_gridsearch(pipeline,param_set=param_set,num_training_images=None)

    # Extract top models
    topModels = get_gridsearch_top_models(gs,numModels=20)
    print("Top validation scores: \n", topModels['test_score'])

    # Save stats to file
    f = save_stats_to_file(gs,topModels,stats,model="RanForest")

    return gs, stats, topModels


gs, stats, topModels = random_forest_pipeline_gridsearch()

