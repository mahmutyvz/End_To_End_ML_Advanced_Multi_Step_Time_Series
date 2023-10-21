from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error, \
    mean_absolute_percentage_error
import numpy as np


def metrics_calculate(y_val,y_pred,X_train):
    """
    Calculate various regression metrics to evaluate the performance of a model.

    Parameters:
    - y_val (pd.Series): Actual target values from the validation set.
    - y_pred (pd.Series): Predicted target values.
    - X_train (pd.DataFrame): Training data features.

    Returns:
    - dict: Dictionary containing the calculated regression metrics (RMSE, MAE, RMSLE, R-Squared, Adj R-Squared, MAPE).
    """
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    rmsle = mean_squared_log_error(y_val, y_pred),
    r2 = r2_score(y_val, y_pred),
    adj_r2 = 1 - (1 - (r2_score(y_val, y_pred))) * (X_train.shape[0] - 1) / (X_train.shape[0] - len(X_train.columns.tolist()) - 1)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    scores = {'RMSE' : rmse,'MAE' : mae,'RMSLE' : rmsle,'R-Squared' : r2,
                  'Adj R-Squared' : adj_r2,'MAPE' : mape}
    return scores
