import numpy as np
from ..Layer import Layer
#Matthew Marmon
class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'TanH Layer'
    def forward(self, dataIn):
        #Input : dataIn , a (1 by K) matrix 
        #Output : A (1 by K) matrix
        self.setPrevIn(np.array(dataIn))
        epsilon = 1e-10
        dataOut = np.zeros_like(dataIn)
        #dataIn = np.clip(dataIn, -50, 50)
        num = np.exp(dataIn)-np.exp(-dataIn)
        denom = np.exp(dataIn) + np.exp(-dataIn) + epsilon
        dataOut = num/denom
        self.setPrevOut(dataOut)
        return dataOut
    
    def gradient(self):
        #Imput: None
        #Output: a N by(K by D) matrix
        # gradient = 1 - g(z)^2
        prev_out = self.getPrevOut()
        grad = 1 - np.square((prev_out))
        N, K = prev_out.shape
        jacobians = np.zeros((N, K, K))
        for i in range(N):
            for j in range(K):
                jacobians[i, j, j] = grad[i, j]
        gradient = jacobians
        return gradient
    def gradient2(self):
        #Imput: none
        #output: a N x K matrix
        out = self.getPrevOut()
        return 1 - np.square(out)
    def backward2(self, dataIn):
        #Input: dataIn, a N by K matrix
        #Output: a N by K matrix
        grad = self.gradient2()
        return dataIn * grad

        