import numpy as np
from ..Layer import Layer
#Matthew Marmon
class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        #Input : dataIn , a (1 by K) matrix 
        #Output : A (1 by K) matrix
        self.setPrevIn(np.array(dataIn))
        dataOut = 1/(1+np.exp(-dataIn))
        self.setPrevOut(dataOut)
        return dataOut
    
    def gradient(self):
        #Imput: None
        #Output: a N by(K by D) matrix
        # gradient = g(z) * (1 - g(z))
        def sigmoid_gradient(z):
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        prev_out = self.getPrevIn()
        grad = sigmoid_gradient(prev_out)
        N, K = prev_out.shape
        jacobians = np.zeros((N, K, K))
        for i in range(N):
            for j in range(K):
                jacobians[i, j, j] = grad[i, j]
        gradient = jacobians
        return gradient
    def gradient2(self):
        #imput: none
        #output: a N x K matrix
        out = self.getPrevOut()
        return out * (1 - out)
    def backward2(self, gradIn):
        #Input: dataIn, a N by K matrix
        #Output: a N by K matrix
        return gradIn * self.gradient2()