import sys
import numpy as np

sys.path.append('../../DL/')
from framework import (
  FullyConnectedLayer, SquaredError, ReLULayer
)

dJdH = np.array([[1.,2.],[4.,5.]],dtype=float)
X = np.array([[1.,2.,3.],[3.,4.,5.]],dtype=float)
eta = 1.0

Y = np.array([[0],[1]],dtype=float)
Yhat = np.array([[0.2],[0.3]],dtype=float)
grad = np.array([[ 0.4], [-1.4]],dtype=float)

try:
    print("Testing ReLU Activation Layer")
    L = ReLULayer()
    
    print("Testing forward method")
    H = np.array([[1.,2.,3.],[3.,4.,5.]],dtype=float)
    
    if(np.all(np.isclose(L.forward(X),H,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", H)
        print("Returned: ", L.forward(X))       

    print("Testing gradient method")
    
    grad = np.array([[[ 1,0,0],[0,1,0],[0,0,1]],[[ 1,0,0],[0,1,0],[0,0,1]]],dtype=float)

    if(np.all(np.isclose(L.gradient(),grad,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", grad)
        print("Returned: ", L.gradient()) 
    print()
except:
    print("Error")
    

try:
    print("Testing SquaredError Objective")
    L = SquaredError()
    
    print("Testing eval method")
    LL = 0.26499999999999996
    
    if(np.isclose(L.eval(Y,Yhat),LL,atol=1e-05)):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", LL)
        print("Returned: ", L.eval(Y,Yhat))       

    print("Testing gradient method")
    
    grad = np.array([[ 0.4], [-1.4]])

    if(np.all(np.isclose(L.gradient(Y,Yhat),grad,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", grad)
        print("Returned: ", L.gradient(Y,Yhat)) 
    print()
except:
    print("Error")
    
try:
    print("Testing FC Layer")
    L = FullyConnectedLayer(3,2)
    L.setPrevIn(X)
    
    L.setWeights(np.array([[1.,2.],[3.,4.],[5.,6.]]))
    L.updateWeights(dJdH,eta)
    What = L.getWeights()

    W = np.array([[-5.5,-6.5],[-6.,-8.],[-6.5,-9.5]])
    
    if(np.all(np.isclose(W,What,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", W)
        print("Returned: ", What)   
    print()
except:
    print("ERROR")









