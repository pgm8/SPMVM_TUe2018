# Rescale data (between 0 and 1)

"""
Question to Rui:
min max normalization is suitable for both positive and negative values?
"""

# Imports

import os
from pickle import dump
from pickle import load

# Load SPX dataframe
picklename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'resources/Data/SP500_data.sav')
spx = load(open(picklename, 'rb'))

# Rescale feature data (between 0 and 1) with lambda function
spx.iloc[:, 1:-1] = spx.iloc[:, 1:-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Save union of normalized feature matrix and response vector with pickle
picklename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'resources/Data/SP500_data_Norm.sav')
dump(spx, open(picklename, 'wb'))

