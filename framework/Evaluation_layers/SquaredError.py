import numpy as np

from ..Layer import Layer
# Matthew Marmon
class SquaredError():
    def eval(self, Y, Yhat):
        # Input: Y, a (N by K) matrix
        # Input: Yhat, a (N by K) matrix
        # Output: a scalar
        return np.mean(np.square(Y - Yhat))
    def gradient(self, Y, Yhat):
        # Input: Y, a (N by K) matrix
        # Input: Yhat, a (N by K) matrix
        # Output: a (N by K) matrix
        gradient = 2 * (Yhat - Y)
        return gradient