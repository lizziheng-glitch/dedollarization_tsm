
import numpy as np
def lag_matrix(y, p):
    y = np.asarray(y, float); T = len(y)
    if T <= p: raise ValueError("Series too short for p lags.")
    Y = y[p:]
    X = np.ones((T-p, p+1))
    for i in range(1, p+1):
        X[:, i] = y[p - i: T - i]
    return Y, X
