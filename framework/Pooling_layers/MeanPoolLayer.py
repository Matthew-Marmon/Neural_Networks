import numpy as np
from ..Layer import Layer
##Matthew Marmon

class MeanPoolLayer(Layer):
    def __init__(self, width, stride):
        #input: width as int, stride as int
        super().__init__()
        self.width = width
        self.stride = stride
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        n, x, y = dataIn.shape
        pool_x = 1+ (x - self.width)//self.stride
        pool_y = 1 + (y - self.width)//self.stride
        pool = np.zeros((n,pool_x,pool_y))
        self.indexes = np.zeros((n,pool_x,pool_y,2))
        for i in range(n):
            for j in range(pool_x):
                for k in range(pool_y):
                    start_x = j * self.stride
                    start_y = k * self.stride
                    end_x = (j+1)*self.width
                    end_y = (j+1)*self.width
                    mapping = dataIn[i , start_x:start_x + self.width , start_y:start_y+self.width]
                    pool[i,j,k] = np.mean(mapping)
                    self.indexes[i,j,k] = [(x,y) for x in range(start_x, end_x) for y in range(start_y, end_y)]
        self.setPrevOut(pool)
        return pool
    def gradient(self):
        pass