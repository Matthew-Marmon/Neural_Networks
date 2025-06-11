import sys,os
import numpy as np
sys.path.append('../../DL/')


from framework import (
  InputLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
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
L4 = LogLoss()
Layers = [L1, L2, L3, L4]
# forward
Yhat = np.zeros((X.shape[0], 1))
H = X
for i in range(len(Layers)-1):
    H = Layers[i].forward(H)
    Yhat = H
#backwards
Yhat = np.atleast_2d(Yhat)
Y = np.atleast_2d(Y).T
grad = Layers[-1].gradient(Y, Yhat)
print("Log loss first observation gradient:", grad[0][0])
grads = []
for i in range(len(Layers)-2, 0, -1):
    grad= Layers[i].backward(grad)
    grads.append(grad)

print("Logisic sigmoid layer first observation gradient:", grads[0][0])
print("Fully connected layer first observation gradient:", grads[1])
