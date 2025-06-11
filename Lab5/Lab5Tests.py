import sys
import numpy as np
sys.path.append('../../DL/')
from framework import (
  TanhLayer
)

H = np.array([[1,2,3],[4,5,6]],dtype=float)

eps = 10**-7

try:
    print("Testing Hyperbolic Tangent Layer")
    L = TanhLayer()
    L.forward(H)
    G = L.gradient()
   
    Y = np.array([[[4.19974342e-01, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 7.06508249e-02, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 9.86603717e-03]],

 [[1.34095068e-03, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 1.81583231e-04, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 2.45765474e-05]]],dtype=float)
    
    if(np.all(np.isclose(G,Y,atol=1e-05))):
        print("Passed")
    else:
        print("Failed.")
        print("Expected: ", Y)
        print("Returned: ", G)   
    print()
except Exception as e:
    print("ERROR:", e)
    print("ERROR")
    
