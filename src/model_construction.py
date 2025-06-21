import pickle
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import joblib 
import os 
from datetime import datetime

def train_model(X_train, y_train):
    """
    Train an XGBoost regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Target values. Can be single or multiple days.

    Returns:
        model: Trained XGBoost model.
    """
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
    model.fit(X_train, y_train)
    return model

def save_model(model, directory='model'):
    """
    Save the trained model to a timestamped .pkl file inside the 'models' directory.

    Args:
        model: Trained machine learning model.
        directory (str): Directory where the model will be saved. Default is "models".
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.pkl"
    filepath = os.path.join(directory, filename)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")
    
def load_model(path='xgboost_predictor.pkl'):
    """
    Load a trained model from a file.

    Args:
        path (str): File path to load the model.

    Returns:
        model: Loaded model.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def multi_step_predict(model, last_known_data, days, feature_creator, selected_columns):
    """
    Predict future values using a trained model for multiple steps.
    
    Args:
        model: Trained machine learning model.
        last_known_data (pd.DataFrame): DataFrame containing the most recent (last day) known data.
        days (int): Number of future days to predict.
        feature_creator (callable): Function to create features from the data.
        selected_columns (list): List of columns used for prediction.
    
    Returns:
        pd.Series: Series containing predicted values for the specified number of days.
    """
    predictions = []
    data = last_known_data.copy()

    for _ in range(days):
        # Recompute features using full data
        features = feature_creator(data)

        # Drop any rows with NaNs (especially important for rolling features)
        features = features.dropna()

        if features.empty:
            predictions.append(np.nan)
            continue

        try:
            # Select last valid row for prediction
            X_input = features[selected_columns].iloc[[-1]]
            y_pred = model.predict(X_input)[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            predictions.append(np.nan)
            continue

        predictions.append(y_pred)

        # Create next day's base row using the most recent known row
        next_row = data.iloc[-1].copy()
        next_row['close'] = y_pred

        # Append the new row to the original data (for next iteration)
        data = pd.concat([data, pd.DataFrame([next_row])], ignore_index=True)

    return pd.Series(predictions)