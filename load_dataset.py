# Module for piling up the preprocessing code required for loading
# the dataset of UCI-HAR.

import pandas as pd
import numpy as np


FEATURES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

MAIN_FOLDER = 'dataset'
SUB_DATA_FOLDER = 'UCI HAR Dataset'

read_data = lambda name : pd.read_csv(name, delim_whitespace=True, header=None)

def get_filename(feature, train):
    """
    Parameters
    ==========

    feature: string
        The feature which is to be read from the corresponding file
    domain: bool
        If True, then read from training files else read from testing files.

    Returns
    =======

    output: string
        File path
    """
    domain = ""
    if train:
        domain = "train"
    else:
        domain = "test"
    if feature is None: # then used for returning target file path
        return f'{MAIN_FOLDER}/{SUB_DATA_FOLDER}/{domain}/y_{domain}.txt'
    return f'{MAIN_FOLDER}/{SUB_DATA_FOLDER}/{domain}/Inertial Signals/{feature}_{domain}.txt'


def get_features(train):
    """
    Read the ``*.txt`` files from the target path of the dataset
    and return numpy feature matrix of appropriate dimension.

    Parameters
    ==========

    train: bool
        If True, the read from training files else read from testing files.

    Returns
    =======

    output: numpy.ndarray
        Returns the feature ``numpy.ndarray`` instance of appropriate dimension
    """
    features = []
    for feat in FEATURES:
        features.append(
            read_data(get_filename(feat, train)).as_matrix()
        )
    return np.transpose(features, (1, 2, 0))

def get_target(train):
    """
    Read the ``*.txt`` files from the target path of the dataset
    and return numpy feature matrix of appropriate dimension (using One-Hot
    encoding).

    Parameters
    ==========

    train: bool
        If true, read from y_train else read from y_test

    Returns
    =======

    output: numpy.ndarray
        Returns the ``numpy.ndarray`` and using one-hot encoding of
        appropriate dimension.
    """
    y = read_data(get_filename(feature=None, train=train))[0]
    return pd.get_dummies(y).as_matrix()

def get_data(train):
    """
    Main driver function for loading the data.

    Parameters
    ==========

    train: bool
        If True, the read from training files else read from testing files.

    Returns
    =======

    output: tuple
        Returns tuple of (X, Y)
    """
    if train:
        return (get_features(train), get_target(train))
    return (get_features(train), get_target(train))
