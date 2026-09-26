import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error


def evaluate_model(actuals, predictions):


    # 1. Get your flattened arrays
    y_true = np.concatenate(actuals).flatten()
    y_pred = np.concatenate(predictions).flatten()

    # 2. Calculate the Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return r2, rmse

