import sys
import numpy as np
sys.path.append('../../DL/')


from framework import (
  InputLayer, FullyConnectedLayer, LogisticSigmoidLayer
)

X = np.array([[-1,2],[0,3]],dtype=float)

x = np.atleast_2d(X[0])

try:
    
    print("Testing Input Layer")
    L = InputLayer(X)
    h = L.forward(x)
    y = np.array([[-0.70710678, -0.70710678]],dtype=float)
    if(np.all(np.isclose(h,y))):
        print("Passed")
    else:
        print("Failed.  Expected: ", y, " returned: ", h)
except:
    print("ERROR")
print()

try:
    print("Testing FC Layer")
    L = FullyConnectedLayer(x.shape[1],3)
    L.setWeights(np.array([[1,2,3],[0,-1,1]],dtype=float))
    L.setBiases(np.array([[1,2,3]],dtype=float))
    h = L.forward(x)
    y = np.array([[ 0, -2,  2]],dtype=float)
    if(np.all(np.isclose(h,y))):
        print("Passed")
    else:
        print("Failed.  Expected: ", y, " returned: ", h)
    
except:
    print("ERROR")
print()


try:
    print("Testing Logistic Sigmoid Layer")
    L = LogisticSigmoidLayer()
    h = L.forward(x)
    y = np.array([[0.26894142, 0.88079708]],dtype=float)
    
    if(np.all(np.isclose(h,y))):
        print("Passed")
    else:
        print("Failed.  Expected: ", y, " returned: ", h)
except:
    print("ERROR")
print()



