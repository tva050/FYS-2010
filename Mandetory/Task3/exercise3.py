import numpy as np
import scipy as sc 
from scipy import signal

w = ([[0,0,1], 
      [0,1,2], 
      [1,2,3]])

f = ([[0,1,0], 
      [0,1,0], 
      [1,0,0]])


grad = sc.signal.convolve2d(w, f, mode='same', boundary='wrap', fillvalue=0)
print(grad)

grad_x = sc.signal.convolve2d(w, f, mode='same', boundary='fill', fillvalue=0)
print(grad_x)

# cyclic convolution