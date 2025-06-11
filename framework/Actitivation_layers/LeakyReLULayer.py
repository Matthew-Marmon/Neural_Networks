# Matthew Marmon 
from ..Layer import Layer
import numpy as np
class LeakyReLULayer(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, dataIn):
        #Input : dataIn , a (1 by K) matrix 
        #Output : A (1 by K) matrix
        self.setPrevIn(np.array(dataIn))
        dataIn = np.array(dataIn)  # ensure it's a NumPy array
        dataOut = np.where(dataIn >= 0, dataIn, dataIn * self.alpha)
        self.setPrevOut(dataOut)
        return dataOut
    def gradient(self):
        out = self.getPrevOut()
        return np.where(out >= 0, 1.0, self.alpha)
    def backward(self,gradIn):
        return gradIn * self.gradient()