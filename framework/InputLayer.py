import numpy as np
from .Layer import Layer
#Matthew Marmon
class InputLayer(Layer):
  def __init__(self, dataIn):
    #input: dataIn as (NxD) matrix
    #Output: none
    super().__init__()
    self.__dataIn = np.array(dataIn)
    self.__meanX = np.mean(self.__dataIn, axis=0)
    self.__stdX = np.std(self.__dataIn, axis=0, ddof=1)
    self.__stdX[self.__stdX == 0] = 1 
  
  def forward(self, dataIn):
    #input: dataIn as (1xD) matrix
    #Output: A (1xD) matrix
    dataIn = np.atleast_2d(np.array(dataIn))
    dataOut = (dataIn - self.__meanX) / self.__stdX
    self.setPrevIn(dataIn)
    self.setPrevOut(dataOut)
    return dataOut
  
  def gradient(self):
    pass
