import numpy as np
from ..Layer import Layer
#Matthew Marmon
class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        #Input : dataIn , a (1 by K) matrix 
        #Output : A (1 by K) matrix
        self.setPrevIn(np.array(dataIn))
        dataOut = np.zeros_like(dataIn)
        dataOut = np.maximum(0, dataIn)
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
                if prev_in[i, j] >= 0:
                    jacobians[i, j, j] = 1.0
        gradient = jacobians
        return gradient 
    def gradient2(self):
        out = self.getPrevOut()
        return (out >= 0).astype(float)
    def backward2(self,gradIn):
        return gradIn * self.gradient2()