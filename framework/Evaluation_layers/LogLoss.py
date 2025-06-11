import numpy as np

from ..Layer import Layer
# Matthew Marmon
class LogLoss():
    def eval(self, Y, Yhat):
        # Input: Y, a (N by K) matrix
        # Input: Yhat, a (N by K) matrix
        # Output: a scalar
        eps = 1e-10
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))
    def gradient(self, Y, Yhat):
        # Input: Y, a (N by K) matrix
        # Input: Yhat, a (N by K) matrix
        # Output: a (N by K) matrix
        eps = 1e-10
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return (Yhat - Y) / (Yhat * (1 - Yhat))