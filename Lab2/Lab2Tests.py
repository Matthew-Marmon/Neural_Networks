import sys
import numpy as np
sys.path.append('../../DL/')

from framework import (
  InputLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
)

H = np.array([[1,2,3],[4,5,6]],dtype=float)

eps = 10**-7

try:
    print("Testing FC Layer")
    L = FullyConnectedLayer(H.shape[0],2)
    L.setWeights(np.array([[1,2],[3,4],[5,6]],dtype=float))
    
    L.setBiases(np.array([[-1,2]],dtype=float))
    L.forward(H)
    G = L.gradient()
    Y = np.array([[[1., 3., 5.], [2, 4, 6]], [[1, 3, 5,], [2, 4, 6,]]],dtype=float)
    
    if(np.all(np.isclose(G[0],Y,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", Y)
        print("Returned: ", G)
    print()
except:
    print("ERROR")


    
try:
    print("Testing Logistic Sigmoid Layer")
    L = LogisticSigmoidLayer()
    L.forward(H)
    G = L.gradient()
    
    Y = np.array([[[0.19661203, 0.,         0. ,       ],
  [0.    ,     0.10499369, 0.        ],
  [0.,         0. ,        0.04517676]],

 [[0.01766281, 0. ,        0.  ,      ],
  [0.  ,       0.00664816, 0.        ],
  [0.   ,      0.     ,    0.00246661]]],dtype=float);
   
    if(np.all(np.isclose(G,Y,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", Y)
        print("Returned: ", G)
    print()
except Exception as e:
    print("ERROR",e)
    


Y = np.array([[0],[1]],dtype=float)
Yhat = np.array([[0.2],[0.3]],dtype=float)


print("Testing Log Loss Objective")
try:
    L = LogLoss()
    
    print("Testing eval method")
    eps = 10**-7
   
    LL = 0.7135579486534379
    
    if(np.isclose(L.eval(Y,Yhat),LL,atol=1e-05)):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", LL)
        print("Returned: ", L.eval(Y,Yhat))
    print()
except:
    print("ERROR")   

try:
    print("Testing gradient method")
    grad = np.array([[ 1.24999984], [-3.33333222]],dtype=float)
    if(np.all(np.isclose(L.gradient(Y,Yhat),grad,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", grad)
        print("Returned: ", L.gradient(Y,Yhat))   
    print()
except:
    print("Error")

