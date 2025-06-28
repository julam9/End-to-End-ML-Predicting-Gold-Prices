from pathlib import Path
import yfinance as yf 
import pandas as pd

def load_gold_data(start_date, end_date, interval, file_name):
    """
    Load gold price data from Yahoo Finance API between start_date and end_date.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        period (str): Period of data to be collected, e.g., '1d', '1wk', '1mo'.

    Returns:
        pd.DataFrames: Two DataFrames consist of gold price data. One for visualization, one for training.  
    """
    try:
        df = yf.download("GC=F", start=start_date, end=end_date, interval=interval)
        # clean the column names
        df.columns = ['_'.join(col).strip("()").split(",")[0].split('_')[0] for col in df.columns]
        # cast the volume data type to integer 
        df['Volume'] = df['Volume'].round().astype(int)
        # save the dataframe as a CSV file 
        df.to_csv(f'data/{file_name}.csv') 
        return df
    
    except Exception as e:
        print(f"API Error : {e} - Attempting local callback" )
        
        try:
            pd.read_csv(f'data/{file_name}.csv', index_col=0, parse_dates=True)
        except Exception as e:
            print(f"File not found: {e}. Please ensure the file exists in the data directory.")
            return None
            