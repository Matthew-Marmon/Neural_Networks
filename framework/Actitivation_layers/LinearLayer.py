import numpy as np
from ..Layer import Layer
#Matthew Marmon
class LinearLayer(Layer):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'Linear Layer'
    def forward(self, dataIn):
        #Input : dataIn , a (1 by K) matrix 
        #Output : A (1 by K) matrix
        self.setPrevIn(np.array(dataIn))
        dataOut = np.zeros_like(dataIn)
        dataOut[:] = dataIn
        self.setPrevOut(dataOut)
        return dataOut
    
    def gradient(self):
        #Imput: None
        #Output: a N by(K by D) matrix
        prev_in = self.getPrevIn()
        N, K = prev_in.shape
        jacobians = np.zeros((N, K, K))
        for i in range(N):
            for j in range(K):
                jacobians[i, j, j] = 1.0
        return jacobians