import os
from pickle import dump


class FeatureNormalizer(object):
    """Feature normalization class. This class has the responsibility to normalize the feature space.
    It does so by transforming all feature values to values in the interval [0, 1]."""

    def __init__(self):
        """Initializer FeatureNormalizer object."""

    def normalize_feature_matrix(self, data):
        """Method for data normalization.
        :param data: set representing the entire data (feature matrix and response vector).
        :return: normalized data set."""
        data.iloc[:, 1:-1] = data.iloc[:, 1:-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return data









