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

w_zeropadded = np.pad(w, 1, 'constant', constant_values=0)
f_zeropadded = np.pad(f, 1, 'constant', constant_values=0)

gra_1 = sc.signal.convolve2d(w_zeropadded, f_zeropadded, mode='same', boundary='wrap', fillvalue=0)
print(gra_1)


# cyclic convolution