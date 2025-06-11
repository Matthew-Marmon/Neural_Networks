import numpy as np

from ..Layer import Layer
# Matthew Marmon
class CrossEntropy():
    def eval(self, Y, Yhat):
        # Input: Y, a (N by K) matrix
        # Input: Yhat, a (N by K) matrix
        # Output: a scalar
        eps = 1e-7
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return -np.sum(Y * np.log(Yhat)) / Y.shape[0]
    
    def gradient(self, Y, Yhat):
        # Input: Y, a (N by K) matrix
        # Input: Yhat, a (N by K) matrix
        # Output: a (N by K) matrix
        eps = 1e-7
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return (-Y / Yhat)