import numpy as np
from ..Layer import Layer
#Matthew Marmon
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return "SoftMax Layer"
    def forward(self, dataIn):
        #Input : dataIn , a (1 by K) matrix 
        #Output : A (1 by K) matrix
        self.setPrevIn(np.array(dataIn))
        
        shifted = dataIn - np.max(dataIn, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.setPrevOut(softmax)
        return softmax
    
    def gradient(self):
        #Input : dataIn , a (n by K) matrix 
        #Output : A (NxKxK) matrix
        dataIn = self.getPrevIn()  # raw logits
        n, k = dataIn.shape
        gradOut = np.zeros((n, k, k))

        
        exp_data = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))  # stability fix
        softmax = exp_data / np.sum(exp_data, axis=1, keepdims=True)

    # Compute Jacobian for each sample
        for i in range(n):
            for j in range(k):
                for l in range(k):
                    if j == l:
                        gradOut[i, j, l] = softmax[i, j] * (1 - softmax[i, j])
                    else:
                        gradOut[i, j, l] = -softmax[i, j] * softmax[i, l]

        return gradOut