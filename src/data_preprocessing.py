from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

def add_variables(df):
    """ 
    Add new features to the DataFrame.

    Args:
        df (pd.DataFrame): Input gold data with its variables like Open, High, Low, Close and other variables.

    Returns:
        pd.DataFrame : Dataframe with new features added.
    """
    
    df["return"] = df["Close"].pct_change()
    df["rolling_mean"] = df["return"].rolling(window=5).mean()
    df["rolling_std"] = df["return"].rolling(window=5).std()
    df["high_low_spread"] = df["High"] - df["Low"]
    df["open_close_spread"] = df["Open"] - df["Close"]
    # df = df.dropna()
    return df

def split_train_test(df, test_size=0.2):
    """
    Splits the data into train and test sets based on time order.

    Args:
        df (pd.DataFrame): The full dataset after feature engineering.
        test_size (float): Fraction of data to reserve for testing.

    Returns:
        pd.DataFrame, pd.DataFrame: train_df, test_df
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def prepare_data_for_training(df, k=3, test_size=0.2):
    """
    Prepare the data for training by splitting and selecting k-best features.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.
        k (int): Number of top features to select.
        test_size (float): Fraction of data to use for testing.

    Returns:
        tuple: X_train, y_train, X_test, y_test, selected_columns
    """
    # Feature columns (must exist after add_variables)
    feature_cols = ["return", "rolling_mean", "rolling_std", "high_low_spread", "open_close_spread"]
    target_col = "Close"

    # Drop rows with missing values just in case
    df = df.dropna(subset=feature_cols + [target_col])

    # Split based on time
    train_df, test_df = split_train_test(df, test_size)

    # Split features/target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_columns = X_train.columns[selector.get_support()]

    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_columns, index=X_test.index)

    return X_train_df, y_train, X_test_df, y_test, selected_columns
