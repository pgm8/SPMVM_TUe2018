import os
from pickle import dump
from pickle import load
from pandas import read_csv
import matplotlib.pyplot as plt

class ModuleManager(object):
    """ModuleManager class. The responsibility of the ModuleManager class is to load, transform and
    save files. If a filename is passed, then this file will be used to load, transform and/or save a
    pickled object (e.g. model or dataset)."""

    def __init__(self, filename=None):
        self.filename = filename if filename else None

    def load_data(self, filename):
        """Load a pickled dataset from the provided filename
        :param filename: filename containing the pickled dataset
        :return: pickled data set"""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            ('resources/Data/%s' % filename))
        return load(open(path, 'rb'))

    def load_model(self, filename):
        """Load a pickled model from the provided filename
        :param filename: filename containing the pickled model
        :return: learner model"""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               'resources/Models/%s' % filename), 'rb') as f:
            return load(f,)

    def save_data(self, filename, data):
        """Save a pickled object to the provided filename
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

    def transform_csv_to_pickle(self, filename):
        """Method to load csv file and save pickled csv file.
        :param filename: filename containing dataset in csv format"""
        csv_file = read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                         ('resources/Data/%s' % filename)), index_col=0)
        self.save_data(filename[:-3]+'pkl', csv_file)

    def transform_pickle_to_csv(self, filename):
        """Method to load a pickle object and save to csv file.
        :param filename: filename containing pickled dataset
        :param data: data set"""
        data_csv = self.load_data(filename)
        filename = filename[:-3]+'csv'
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            ('resources/Data/%s' % filename))
        data_csv.to_csv(path)

    def load_csv(self, filename):
        return (read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                         ('resources/Data/%s' % filename)), index_col=0))

    def save_fig(self, figure, fig_name):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                                            ), ('LaTex/figures/multivar_analysis/%s' % fig_name))
        figure.savefig(path, format='pdf')




