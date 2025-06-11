import numpy as np
from ..Layer import Layer
##Matthew Marmon

class MaxPoolLayer(Layer):
    def __init__(self, width, stride):
        #input: width as int, stride as int
        super().__init__()
        self.width = width
        self.stride = stride
    def __str__(self):
        return 'Max Pool Layer'
    def forward(self, dataIn):
        #input: Data as a NxKxD tensor
        #output: pooled data as a N x floor(K-width/stride) +1 x floor(D-width/stride) +1 tensor
        self.setPrevIn(dataIn)
        self.__shape = dataIn.shape
        n, x, y = dataIn.shape
        pool_x = 1+ (x - self.width)//self.stride
        pool_y = 1 + (y - self.width)//self.stride
        pool = np.zeros((n,pool_x,pool_y))
        self.max_indexes = np.zeros((n,pool_x,pool_y,2))
        for i in range(n):
            for j in range(pool_x):
                for k in range(pool_y):
                    start_x = j * self.stride
                    start_y = k * self.stride
                    mapping = dataIn[i , start_x:start_x + self.width , start_y:start_y+self.width]
                    max_index = np.unravel_index(np.argmax(mapping),mapping.shape)
                    pool[i,j,k] = mapping[max_index]
                    self.max_indexes[i,j,k] = (start_x + max_index[0],start_y + max_index[1])
        self.setPrevOut(pool)
        return pool
    def backward(self, gradIn):
        
        self.setPrevIn(gradIn)
        return self.gradient()
    def gradient(self):
        gradIn = self.getPrevIn()
        n, x, y = self.__shape
        backprop = np.zeros((n, x, y))
        _, x_pool, y_pool = gradIn.shape
        for i in range(n):
            for j in range(x_pool):
                for k in range(y_pool):
                    idx_x = int(self.max_indexes[i, j, k, 0])
                    idx_y = int(self.max_indexes[i, j, k, 1])
                    backprop[i, idx_x, idx_y] = gradIn[i, j, k]
        self.setPrevOut(backprop)
        return backprop
    
    