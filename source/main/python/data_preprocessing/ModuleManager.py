import os
from pickle import dump
from pickle import load


class ModuleManager(object):
    """ModuleManager class. The responsibility of the ModuleManager class is to load and
    save files from and to a specified disk location. If a filename is passed,
    then this file will be used to save or load a pickled object (i.e. model or dataset) to or from,
    respectively, a provided disk location.
    """

    def __init__(self, filename=None):
        self.filename = filename if filename else None

    def load_data(self, filename):
        """Load a pickled dataset from the provided filename
        :param filename: filename containing the pickled dataset
        :return: pickled data set"""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Data/%s' % filename),'rb') as f:
            return load(f)

    def load_model(self, filename):
        """Load a pickled model from the provided filename
        :param filename: filename containing the pickled model
        :return: learner model"""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Models/%s' % filename),'rb') as f:
            return load(f)

    def save_data(self, filename, data):
        """Save a pickled object to the provied filename
        :param filename: filename containing the pickled dataset
        :param data: data set"""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           ('resources/Data/%s' % filename))
        dump(data, open(path, 'wb'))

    def save_model(self, filename, model):
        """Save a pickled model to the provided filename
        :param filename: filename containing the pickled model
        :param model: learner model"""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            ('resources/Models/%s' % filename))
        dump(model, open(path, 'wb'))



