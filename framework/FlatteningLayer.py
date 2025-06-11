import numpy as np
from .Layer import Layer
##Matthew Marmon

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'Flattening Layer'
    def forward(self,prevOut):
        self.setPrevIn(prevOut)
        self.__shape = prevOut.shape
        n = prevOut.shape[0]
        out = np.reshape(prevOut,(n,-1) ,order='F')
        self.setPrevIn(out)
        return out
    def backward(self, gradIn):
        self.setPrevIn(gradIn)
        return self.gradient()
    def gradient(self):
        prevIn = self.getPrevIn()
        gradOut = np.reshape(prevIn, self.__shape, order='F')
        self.setPrevOut(gradOut)
        return gradOut