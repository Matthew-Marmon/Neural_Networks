import numpy as np
from .Layer import Layer
##Matthew Marmon

class ConvolutionalLayer(Layer):
    def __init__(self, sizeIn, method = '2D'):
        super().__init__()
        self.size = sizeIn
        self.__method = method
        if method == '2D':
            self.__kernel = np.random.uniform(-1e-4,1e-4, (sizeIn, sizeIn))
        elif method == '3D':
            self.__kernel = np.random.uniform(-1e-4,1e-4, (sizeIn, sizeIn, sizeIn))
        elif method == '1D':
            self.__kernel = np.random.uniform(1e-4,1e-4,(1, sizeIn))
        else:
            raise ValueError('Invalid method. Use "1D", "2D" or "3D".')
    def __str__(self):
        return 'Convolutional Layer'
    def setKernels(self, kernel):
        self.__kernel = kernel
    def getKernels(self):
        return self.__kernel
    
    @staticmethod
    def crossCorrelate2D(kernel, dataIn):
        #cross correlation of two matrixes, kernel first, then dataIn
        r,c = dataIn.shape
        x = kernel.shape[0] #assumes square
        r_out = r-x+1
        c_out = c-x+1
        cross = np.zeros((r_out, c_out))
        for i in range(r_out):
            for j in range(c_out):
                data = dataIn[i:i+x,j:j+x]
                cross[i,j] = np.sum(data * kernel)
        return cross

    @staticmethod
    def crossCorrelate3D(kernel, dataIn):
        pass

    @staticmethod 
    def crossCorrelate1D(kernel,dataIn):
        pass

    def forward(self, dataIn):
        #input: NxHxW tensor
        #Output: tensor
        self.setPrevIn(dataIn)
        if self.__method == '2D':
            n = dataIn.shape[0]
            h_out = dataIn.shape[1] - self.size + 1
            w_out = dataIn.shape[2] - self.size + 1
            tensor = np.zeros((n, h_out, w_out))
            for i in range(n):
                tensor[i] = ConvolutionalLayer.crossCorrelate2D(self.__kernel,dataIn[i])
            self.setPrevOut(tensor)
        elif self.__method == '1D':
            pass
        elif self.__method == '3D':
            pass
        return tensor

    def gradient(self, gradIn):
        # only 2d for now
        #djdx = zpad(gradIn) *star rot180(kernel)
        kernel = self.getKernels()
        flipped = np.rot90(kernel, 2)
        n, h_out, w_out = gradIn.shape
        k = kernel.shape[0]
        h_in = h_out + k - 1
        w_in = w_out + k - 1
        djdx = np.zeros((n, h_in, w_in))
        for i in range(n):
            for j in range(h_out):
                for l in range(w_out):
                    djdx[i, j:j+k, l:l+k] += gradIn[i, j, l] * flipped
        return djdx

    def backward(self, gradIn):
        return self.gradient(gradIn)

    def updateKernels(self, gradIn, learningRate):
        #input: gradient in and learning rate as possitive float
        #output: updated kernel
        ##only 2d for now
        prevIn = self.getPrevIn()  
        kernel = self.getKernels()
        djdk = np.zeros_like(self.__kernel)
        n, h_out, w_out = gradIn.shape
        k = self.__kernel.shape[0]

        for i in range(n):
            for j in range(h_out):
                for l in range(w_out):
                    region = prevIn[i, j:j+k, l:l+k]
                    djdk += gradIn[i, j, l] * region
        kernel =kernel-(learningRate * djdk)
        self.setKernels(kernel)
        return kernel