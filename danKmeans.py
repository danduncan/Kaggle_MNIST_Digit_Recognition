# My implementation of k means as a supervised learning algorithm

# For each class, find N centroids of that class using k means
# Train k neareast neighbors on this set of centroids and use it to classify future data

from getTrainingData import *
import time  # Allow time.time()
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Function to create and train Kmeans->KNN pipeline
def hybrid_k(X,y,n_clusters=8,n_neighbors=2,n_jobs=-1):
    # Get all unique class labels
    labels = np.unique(y)
    numLabels = len(labels)
    numFeatures = X.shape[1]

    # Create list of lists, where each sub-list will hold data for a particular label
    t = y.as_matrix().ravel()
    x = [[] for xx in range(numLabels)]

    # Separate all data by class
    for i in range(numLabels):
        x[i] = X[t == labels[i]]

    # Perform k means on each class subset and get centroids
    centroids = np.empty([numLabels * n_clusters, numFeatures])
    centroidLabels = np.zeros([numLabels * n_clusters])
    tstart = time.time()
    for i in range(numLabels):
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit(x[i])
        startIndex = i * n_clusters
        endIndex = startIndex + n_clusters
        centroids[startIndex:endIndex, :] = kmeans.cluster_centers_
        centroidLabels[startIndex:endIndex] = centroidLabels[startIndex:endIndex] + labels[i]
    tkmeans = time.time() - tstart
    print("Kmeans train time:  ", "%.3f" % tkmeans, "s")

    # Create K nearest neighbors classifier and train
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    tstart = time.time()
    clf.fit(centroids, centroidLabels)
    tknn = time.time() - tstart
    print("KNN train time:  ", "%.3f" % tknn, "s")

    tstart = time.time()
    score0 = clf.score(train_images, train_labels)
    tval = time.time() - tstart
    print("Validation time:  ", "%.3f" % tval, "s")
    print("Training accuracy: ", "%.2f" % (100 * score0), '%\n')
    return clf



# Create class for classifier
# Using "object" makes HybridKMeans a subclass of the generic object class
class HybridKMeans(object):
    def __init__(self, n_clusters=24, n_neighbors=2):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.n_jobs = -1
        self.centroids = None
        self.centroidLabels = None
        self.classifier = None

    def setCentroids(self,X,y):
        # Get all unique class labels
        labels = np.unique(y)
        numLabels = len(labels)
        numFeatures = X.shape[1]

        # Separate data by class and perform k means on each subset to get centroids
        centroids = np.empty([numLabels * self.n_clusters, numFeatures])
        centroidLabels = np.zeros([numLabels * self.n_clusters])
        for i in range(numLabels):
            x = X[y == labels[i]]
            kmeans = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(x)
            startIndex = i * self.n_clusters
            endIndex = startIndex + self.n_clusters
            centroids[startIndex:endIndex, :] = kmeans.cluster_centers_
            centroidLabels[startIndex:endIndex] = centroidLabels[startIndex:endIndex] + labels[i]

        self.centroids = centroids
        self.centroidLabels = centroidLabels
        return

    def setKNN(self):
        # Create K nearest neighbors classifier and train
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors).fit(self.centroids, self.centroidLabels)

    def fit(self,X,y):
        self.setCentroids(X,y)
        self.setKNN()

    def predict(self,X):
        return self.classifier.predict(X)

    def score(self,X,y):
        return self.classifier.score(X,y)

    def get_params(self,deep=True):
        return {'n_clusters': self.n_clusters, 'n_neighbors': self.n_neighbors}

    def set_params(self,n_clusters=24, n_neighbors=2):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        return


def main():
    # Set parameters
    NUM_TRAINING_IMAGES = 40000
    n_clusters = 24
    n_neighbors = 4

    # Get the training data
    binarizeData = True
    scaleData = False
    pca_dimension = None
    train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES,
                                                                             binarizeData,
                                                                             scaleData,
                                                                             pca_dimension)

    # Get trained classifier
    print("HybridKMeans Class:")
    clf = HybridKMeans(n_clusters=n_clusters, n_neighbors=n_neighbors)


    tstart = time.time()
    clf.fit(train_images, train_labels.as_matrix().ravel())
    ttrain = time.time() - tstart
    print("Train time:      ", "%.3f" % ttrain, "s")

    t0 = time.time()
    score1 = clf.score(train_images,train_labels)
    tval = time.time() - t0
    print("Predict time:    ", "%.3f" % tval, "s")

    # Validate classifier
    t0 = time.time()
    score2 = clf2.score(test_images,test_labels)
    ttest = time.time() - t0
    ttotal = time.time() - tstart
    print("Validation time: ", "%.3f" % ttest, "s")
    #print("Total time:      ", "%.3f" % ttotal, "s")
    print("Training accuracy:   ", "%.2f" % (100*score1), '%')
    print("Validation accuracy: ", "%.2f" % (100*score2), '%\n')


def mainCV():
    # Run grid search on HybridKMeans
    params = {'n_clusters': [1,2],
              'n_neighbors': [1,2]}
    refit = True


    NUM_TRAINING_IMAGES = 500
    # Get the training data
    binarizeData = True
    scaleData = False
    pca_dimension = None
    train_images, train_labels, test_images, test_labels = get_training_data(NUM_TRAINING_IMAGES,
                                                                             binarizeData,
                                                                             scaleData,
                                                                             pca_dimension)

    train_labels = train_labels.as_matrix().ravel()

    clf = GridSearchCV(HybridKMeans(), params)


    t0 = time.time()
    clf.fit(train_images,train_labels)
    tgrid = time.time() - t0
    print("Total search time: ", "%.3f" % tgrid, "s")

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

mainCV()