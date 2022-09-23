

import numpy as np

def fct_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100