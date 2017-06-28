import os
from pickle import dump


class FeatureNormalizer(object):
    """Feature normalization class. This class has the responsibility to normalize the feature space.
    It does so by transforming all feature values to values in the interval [0, 1]."""

    def __init__(self, data):
        """Initializer FeatureNormalizer object.
        :param data: set representing the entire data (feature matrix and response vector)."""
        self.data = data
        self.normalizeFeatureMatrix(data)

    def normalizeFeatureMatrix(self, data):
        """Method for data normalization.
        :param data: set representing the entire data (feature matrix and response vector).
        :return: normalized data set."""
        data.iloc[:, 1:-1] = data.iloc[:, 1:-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return data

    def saveNormalizedData(self, data, filename):
        """Method for writing normalized data to resource directory.
        :param data: normalized data set
        :param filename: name of object containing normalized data set."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ('resources/Data/%s'
                                                                                          % filename))
        dump(data, open(path, 'wb'))








