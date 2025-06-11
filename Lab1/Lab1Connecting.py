import sys,os
import numpy as np
sys.path.append('../../DL/')


from framework import (
  InputLayer, FullyConnectedLayer, LogisticSigmoidLayer
)
#load data from csv
file_path = os.path.join(os.getcwd(), 'KidCreative.csv')
X = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=float)
Y = X[:,1]
X = X[:, 2:]

#layers 
L1 = InputLayer(X)
L2= FullyConnectedLayer(X.shape[1], np.atleast_2d(Y).T.shape[1])
L3 = LogisticSigmoidLayer()
Layers = [L1, L2, L3]
# forward
Yhat = np.zeros((X.shape[0], 1))
for i,x in enumerate(X):
    for L in Layers:
        x = L.forward(x)
    Yhat[i] = x
# print first observation's prediction
print("First observation's prediction:", Yhat[0][0])