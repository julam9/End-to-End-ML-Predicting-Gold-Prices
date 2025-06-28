import yfinance as yf
import pandas as pd

df_training = yf.download("GC=F", start='2022-01-01', end='2024-12-31', interval='1d')
# clean the column names
df_training.columns = ['_'.join(col).strip("()").split(",")[0].split('_')[0] for col in df_training.columns]
# cast the column type to float
df_training['Volume'] = df_training['Volume'].round().astype(int)
# save the dataframe as a CSV file 
df_training.to_csv('data/data_training.csv')

df_visualization = yf.download("GC=F", start='2015-01-01', end='2024-12-31', interval='1d')
# clean the column names
df_visualization.columns = ['_'.join(col).strip("()").split(",")[0].split('_')[0] for col in df_visualization.columns]
# cast the column type to float
df_visualization['Volume'] = df_visualization['Volume'].round().astype(int)
# save the dataframe as a CSV file 
df_visualization.to_csv('data/data_visualization.csv')


