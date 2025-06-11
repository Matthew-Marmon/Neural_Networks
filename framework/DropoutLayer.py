import numpy as np
from .Layer import Layer
##Matthew Marmon

class DropoutLayer(Layer):
    #####  make sure to exclude from any testing or validation sets #######
    def __init__(self, prob):
        #dropout probability 
        super().__init__
        self.prob = prob
        self.mask = None
    def forward(self, dataIn):
        #Input: dataIn
        self.setPrevIn = dataIn
        self.mask = (np.random.rand(*dataIn.shape)> self.prob).astype(dataIn.dtype)
        out = dataIn * self.mask / (1-self.prob)
        self.setPrevOut = out
        return out
    def backward(self, gradIn):
        return self.mask * gradIn /(1-self.prob)
    def gradient(self, gradIn):
        return self.backward(gradIn)