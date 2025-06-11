import numpy as np
np.random.seed(0) #for reproducibility, change as needed
from .Layer import Layer
# Matthew Marmon
class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.__weights = np.random.uniform(-1e-4,1e-4, (sizeIn, sizeOut))
        self.__biases = np.random.uniform(-1e-4,1e-4, (1, sizeOut))

        #adam initialization
        self.m_w = np.zeros_like(self.__weights)
        self.v_w = np.zeros_like(self.__weights)
        self.m_b = np.zeros_like(self.__biases)
        self.v_b = np.zeros_like(self.__biases)
        self.t = 0
        
    def __str__(self):
        return 'Fully Connected Layer'

    def getWeights(self):
        return self.__weights
    def setWeights(self, weights):
        self.__weights = weights
        
    def updateWeights(self, GradIn, learningRate):
        N = GradIn.shape[0]
        dJdb = np.sum(GradIn, axis=0) / N
        dJdW = (self.getPrevIn().T @ GradIn) / N
        self.__weights -= learningRate * dJdW
        self.__biases -= learningRate * dJdb

    def updateWeights2(self, GradIn, learningRate, beta1= 0.9, beta2= 0.999, epsilon= 1e-8):
        # Update weights using Adam optimization
        N = GradIn.shape[0]
        dJdb = np.sum(GradIn, axis=0) / N
        dJdW = (self.getPrevIn().T @ GradIn) / N
        self.t += 1

        self.m_w = beta1 * self.m_w + (1 - beta1) * dJdW
        self.v_w = beta2 * self.v_w + (1 - beta2) * (dJdW ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * dJdb
        self.v_b = beta2 * self.v_b + (1 - beta2) * (dJdb ** 2)

        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)

        self.__weights -= learningRate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.__biases -= learningRate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def getBiases(self):
        return self.__biases
    def setBiases(self, biases):    
        self.__biases = biases
    def setPrevIn(self, dataIn):
        # Store the input data for use in backpropagation
        self.__prevIn = dataIn

    def getPrevIn(self):
        # Retrieve the stored input data
        return self.__prevIn

    def forward(self, dataIn):
        # input: dataIn as (1xD) matrix
        # Output: A (1xM) matrix where M is the output size
        self.setPrevIn(dataIn)
        dataOut = np.dot(dataIn, self.__weights) + self.__biases
        self.setPrevOut(dataOut)
        return dataOut
    def gradient(self):
        #Imput: None
        #Output: a N by(K by D) matrix
        numb = self.getPrevIn().shape[0]
        weights = self.getWeights()
        return np.tile(weights.T, (numb, 1, 1))
    def backward2(self, gradIn):
        return np.dot(gradIn, self.__weights.T)