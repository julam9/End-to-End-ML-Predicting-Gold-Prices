from sklearn.metrics import mean_absolute_error, root_mean_squared_error
def evaluate_model(y_true, y_pred):
    """
    Evaluation of model performance using two metrics : Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    Args:
        y_true (array): the close price of gold
        y_pred (array): the predicted close price of gold

    Returns:
        array : array that consists of mean absolute error and root mean squared error
    """
    model_mae = mean_absolute_error(y_true, y_pred)
    model_rmse = root_mean_squared_error(y_true, y_pred)
    return {"MAE" : model_mae, "RMSE" : model_rmse}