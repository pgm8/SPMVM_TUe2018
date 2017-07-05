from pandas_datareader import data as dt
from ModuleManager import ModuleManager
from TechnicalAnalyzer import TechnicalAnalyzer
from FeatureNormalizer import FeatureNormalizer


def data_retrieval():
    """Method to pull financial data from yahoo.finance. Note: API does not seem to work anymore
    due to changes of webpage structure at server's end."""
    tickers = ['SPY']  # ETF SPY as proxy for S&P 500
    data_source = 'google'
    start = '1992-01-01'
    end = '2016-12-31'
    data_panel = dt.DataReader(tickers, data_source=data_source, start=start, end=end)
    print(type(data_panel))  # Weird ass data structure: panel


def main():
    mm = ModuleManager()
    ta = TechnicalAnalyzer()
    ft = FeatureNormalizer()

    filenames = ['GSPC.csv']
    for filename in filenames:
        # Transform csv files to pickled objects
        mm.transform_csv_to_pickle(filename)
        # Load pickled files
        data = mm.load_data(filename[:-4] + '_csv.pkl')
        # Feature Engineering and Normalization:
        # Create pickled dataframes obtained form technical analysis
        # 1a) base: price and volume data + return + response
        data_ta_base = ta.ta_base(data)
        mm.save_data(filename[:-4] + '_base.pkl', data_ta_base)
        # 1b) normalize base dataset:
        data_base_normalized = ft.normalize_feature_matrix(data_ta_base)
        mm.save_data(filename[:-4] + '_base_norm.pkl', data_base_normalized)
        # 2a) full: price and volume data + feature set + return + response
        data_ta_full = ta.ta_full(data)
        mm.save_data(filename[:-4] + '_full.pkl', data_ta_full)
        # 2b) normalize full dataset:
        data_full_normalized = ft.normalize_feature_matrix(data_ta_full)
        mm.save_data(filename[:-4] + '_full_norm.pkl', data_full_normalized)


if __name__ == '__main__':
    main()
