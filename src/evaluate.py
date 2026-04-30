from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def adjusted_r2(X, y, y_pred):
    r2 = r2_score(y, y_pred)
    n = X.shape[0] # Number of observatiions
    p = X.shape[1] # Number of features
    if n <= p + 1:
        return None
    else: return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mse(y, y_pred):
    return mean_squared_error(y, y_pred)