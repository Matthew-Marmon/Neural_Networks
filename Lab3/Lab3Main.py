import sys,os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../DL/')
#Matthew Marmon

from framework import (
  InputLayer, FullyConnectedLayer, ReLULayer, SquaredError
)
#load data from csv
filepath = os.path.join(os.getcwd(), 'medical.csv')
X = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=float)
# shuffle rows
np.random.shuffle(X)
#set x and y for training and testing sets
Y = X[:,-1]
X = X[:, :-1]
Ytrain = Y[:int(0.66*len(Y))]
Xtrain = X[:int(0.66*len(X))]
Ytest = Y[int(0.66*len(Y)):]
Xtest = X[int(0.66*len(X)):]
SME_train, SME_test = [], []
#layers = []
L1 = InputLayer(Xtrain)
L2= FullyConnectedLayer(X.shape[1], 1)
L3 = ReLULayer()
L4 = SquaredError()
Layers = [L1, L2, L3]
max_epoch = 1000
epoch = 0 
eta = 1e-3
avg = 1
# training
while epoch < max_epoch and avg > 1e-10:
    epoch += 1
    H = Xtrain
    for i in range(len(Layers)):
        H = Layers[i].forward(H)
    Yhat = H
    Yhat = np.atleast_2d(Yhat)
    Ytrain = np.atleast_2d(Ytrain).T
    SME_train.append(L4.eval(Ytrain, Yhat))
    grad = L4.gradient(Ytrain, Yhat)
    for i in range(len(Layers)-1, 0, -1):
        new_grad = Layers[i].backward(grad)
        if isinstance(Layers[i], FullyConnectedLayer):
            Layers[i].updateWeights(grad, eta)
        grad = new_grad
    H_test = Xtest
    for i in range(len(Layers)):
        H_test = Layers[i].forward(H_test)
    Yhat_test = H_test
    Yhat_test = np.atleast_2d(Yhat_test)
    Ytest = np.atleast_2d(Ytest).T
    SME_test.append(L4.eval(Ytest, Yhat_test))
    if len(SME_train) > 1:
        avg = np.mean(np.abs(SME_train[-1] - SME_train[-2]))
    else:
        avg = float('inf')  # Set avg to a high value initially
    Ytrain = np.atleast_2d(Ytrain).T
    Ytest = np.atleast_2d(Ytest).T
print(f'Final Epoch: {epoch}')
train_SMAPE = np.mean(2*abs(Ytrain - Yhat) / (abs(Ytrain) + abs(Yhat)))*100
test_SMAPE = np.mean(2*abs(Ytest - Yhat_test) / (abs(Ytest) + abs(Yhat_test)))*100
train_RMSE = np.sqrt(np.mean((Ytrain - Yhat)**2))
test_RMSE = np.sqrt(np.mean((Ytest - Yhat_test)**2))
print(f'Training SMAPE: {train_SMAPE:.4f}')
print(f'Testing SMAPE: {test_SMAPE:.4f}')
print(f'Training RMSE: {train_RMSE:.4f}')
print(f'Testing RMSE: {test_RMSE:.4f}')
plt.title('Squared Error Loss')
plt.plot(SME_test, label='Test Loss')
plt.plot(SME_train, label='Train Loss')
plt.legend()
plt.xlabel('EPOCH')
plt.ylabel('Squared Error Loss')
plt.show()