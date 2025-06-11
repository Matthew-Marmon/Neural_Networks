# matthew marmon
import sys, os
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.append('../../DL/')
from framework import (InputLayer, 
                       FullyConnectedLayer, 
                       LogisticSigmoidLayer, 
                       LogLoss,
                       TanhLayer)


file_path = os.path.join(os.getcwd(), 'KidCreative.csv')
X = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=float)
#shuffle rows
np.random.shuffle(X)
Y = X[:,1]
X = X[:, 2:]

Ytrain = Y[:int(0.66*len(Y))]
Xtrain = X[:int(0.66*len(X))]
Ytest = Y[int(0.66*len(Y)):]
Xtest = X[int(0.66*len(X)):]

imput_dim = Xtrain.shape[1]
hidden_dim = Xtrain.shape[1] // 2

L1 = InputLayer(Xtrain)
L2= FullyConnectedLayer(imput_dim, hidden_dim)
L3 = TanhLayer()
L4 = FullyConnectedLayer(hidden_dim, 1)
L5 = LogisticSigmoidLayer()
L6 = LogLoss()
Layers = [L1, L2, L3,L4,L5,L6]


#he weight init
init1 = np.sqrt(2 / imput_dim)
init2 = np.sqrt(2 / hidden_dim)
L2.setWeights(np.random.uniform(-init1, init1, (imput_dim, hidden_dim)))
L2.setBiases(np.random.uniform(-init1, init1, (1, hidden_dim)))
L4.setWeights(np.random.uniform(-init2, init2, (hidden_dim, 1)))
L4.setBiases(np.random.uniform(-init2, init2, (1, 1)))

ll_train, ll_test = [],[]
max_epoch = 10000
epoch = 0 
eta = 5e-4


tic = time.time()
while epoch < max_epoch:
    epoch += 1 
    #forward
    H = Layers[0].forward(Xtrain)
    for L in range(1, len(Layers)-1):
        H = Layers[L].forward(H)
    Yhat = H
    Yhat = np.atleast_2d(Yhat)
    Ytrain = np.atleast_2d(Ytrain).T
    ll = Layers[-1].eval(Ytrain, Yhat)
    ll_train.append(ll)
    #backward
    grad = Layers[-1].gradient(Ytrain, Yhat)
    for L in range(len(Layers)-2, 0, -1):
        if isinstance(Layers[L], FullyConnectedLayer):
            Layers[L].updateWeights2(grad, eta)
        grad = Layers[L].backward(grad)
    #test
    H = Layers[0].forward(Xtest)  # Normalize test data using training stats
    for L in range(1, len(Layers)-1):
        H = Layers[L].forward(H)
    Yhat_test = H
    Yhat_test = np.atleast_2d(Yhat_test)
    Ytest = np.atleast_2d(Ytest).T
    ll_test.append(Layers[-1].eval(Ytest, Yhat_test))
    if len(ll_train) > 1:
        avg = np.mean(np.abs(ll_train[-1] - ll_train[-2]))
    else:
        avg = float('inf')  # Set avg to a high value initially
    Ytrain = np.atleast_2d(Ytrain).T
    Ytest = np.atleast_2d(Ytest).T
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training Log Loss: {ll_train[-1]:.4f}, Test Log Loss: {ll_test[-1]:.4f}")
toc = time.time()
print(f'Final Epoch: {epoch}, Time Taken: {toc - tic:.2f} seconds')
# Plotting the log loss
plt.figure(figsize=(10, 5))
plt.plot(ll_train, label='Training Log Loss', color='blue')
plt.plot(ll_test, label='Test Log Loss', color='orange')
plt.title('Log Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.grid()
plt.show()
#accuracies
Yhat = (Yhat >= 0.5).astype(int)
Yhat_test = (Yhat_test >= 0.5).astype(int)
train_accuracy = np.mean(Yhat == Ytrain) * 100
test_accuracy = np.mean(Yhat_test == Ytest) * 100
print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Testing Accuracy: {test_accuracy:.2f}%')
'''
just some observations 
1. extreme overfitting
2. low accuracy 
3. speed up is dramatic
4. good log loss but idk whats up with that compared to accuracies 


'''
