import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os
from alpha_vantage.timeseries import TimeSeries

# Construct path to .env in the parent directory
env_path = Path('..') / '.env'

# Load the .env file
load_dotenv(dotenv_path=env_path)

def load_gold_data(start_date=None, end_date=None, api_key=None):
    """
    Load gold price data from Alpha Vantage API between start_date and end_date.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str, optional): Your Alpha Vantage API key. If None, loads from .env.

    Returns:
        pd.DataFrames: Two DataFrames consist of gold price data. One for visualization, one for training.
    """
    if api_key is None:
        api_key = os.getenv("ALPVAN_API_KEY")

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol='GLD', outputsize='full') 
    # set datetime as index
    data.index = pd.to_datetime(data.index)
    # tidying up the column names
    data.columns = data.columns.str.replace(r'^\d+\.\s*', '', regex=True)
    # sort the index so we can slice it
    data = data.sort_index()
    # condition for start and end date
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]
    # filter the data to use
    df = data[start_date:end_date]
    # change volume into integer data type
    df['volume'] = df['volume'].round().astype(int)
    return df