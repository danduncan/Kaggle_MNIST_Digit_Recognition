# Function to import MNIST training data

import pandas as pd
from sklearn.model_selection import train_test_split
import copy  # Using copy.deepcopy()


def get_training_data(num_training_images, binarize_data):
    # Images used are from MNIST dataset: 28x28 pixels
    image_dim = 28

    # Use read_csv to read training file into a dataframe
    labeled_images = pd.read_csv('./input/train.csv')

    # Show how many images are in dataset
    # Each image is 785 elements: 1 label, then 28*28 = 784 pixels
    print(labeled_images.size / (image_dim ** 2 + 1), 'total images in training set.')

    # iloc selects data based on its integer position in the array
    # Alternatively, can use loc to select data by label type
    images = labeled_images.iloc[0:num_training_images, 1:]
    labels = labeled_images.iloc[0:num_training_images, :1]

    # Output how many images are actually being used
    print(images.size / (image_dim ** 2), 'images used for training.')

    # Split training set into training and validation sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,
                                                                            random_state=0)

    # Not that the split above does not copy the data
    # It merely creates a reference to the original. Need to manually deep copy:
    train_images = copy.deepcopy(train_images)
    test_images = copy.deepcopy(test_images)
    train_labels = copy.deepcopy(train_labels)
    test_labels = copy.deepcopy(test_labels)

    # Clean up the data by rounding real values to either 0 or 1
    if binarize_data:
        test_images[test_images > 0] = 1
        train_images[train_images > 0] = 1

    return train_images, train_labels, test_images, test_labels
