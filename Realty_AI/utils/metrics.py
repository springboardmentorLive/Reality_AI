import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates MAE and RMSE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return mae, rmse
