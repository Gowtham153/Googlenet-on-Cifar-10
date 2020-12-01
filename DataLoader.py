import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_train = None
    y_train = []

    for i in range(1, 6):
        file=data_dir + "/data_batch_{}".format(i)
        with open(file, 'rb') as fo:
            data_dic = pickle.load(fo, encoding='latin1')
            
        if i == 1:
            x_train = data_dic['data']
        else:
            x_train = np.vstack((x_train, data_dic['data']))
        y_train += data_dic['labels']

    x_train = np.array(x_train, dtype=np.float32)
    #print(x_train.shape)
    x_train = x_train.reshape(50000,3072)
    y_train = np.array(y_train, dtype=np.int32)
    y_train = y_train.reshape(50000,)

    file=data_dir + "/test_batch"
    with open(file, 'rb') as fo:
        test_data_dic = pickle.load(fo, encoding='latin1')
    x_test = test_data_dic['data']
    y_test = test_data_dic['labels']
     
    x_test = np.array(x_test, dtype=np.float32)
    x_test = x_test.reshape(10000,3072)
    y_test = np.array(y_test, dtype=np.int32)
    y_test = y_test.reshape(10000,)
    x_train = np.concatenate((x_train,x_test), axis=0)
    y_train = np.concatenate((y_train,y_test), axis=0)
    
    ### END CODE HERE

    return x_train, y_train


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    x_test = np.load(data_dir)
    x_test = np.array(x_test, dtype=np.float32)
    
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=5/6):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index=int(train_ratio* len(x_train))
    x_train_new=x_train[:split_index]
    y_train_new=y_train[:split_index]
    x_valid=x_train[split_index:]
    y_valid=y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

