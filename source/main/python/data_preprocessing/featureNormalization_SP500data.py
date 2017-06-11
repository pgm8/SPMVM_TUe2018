# Rescale data (between 0 and 1)

"""
Question to Rui:
min max normalization is suitable for both positive and negative values?
"""

# Imports
from pandas import read_csv
import os

# Load CSV into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/SP500_data.csv')
spx = read_csv(filename)
spx = spx.drop('Unnamed: 0', axis=1)


# Rescale feature data (between 0 and 1) with lambda function
spx.iloc[:, 1:-1] = spx.iloc[:, 1:-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# Write normalized feature data into CSV file
spx.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/SP500_data_Norm.csv'))

