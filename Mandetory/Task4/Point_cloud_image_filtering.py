import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndi
import scipy as sc
from scipy.signal import wiener

plt.style.use('ggplot')

"""  ------------------ 1. Load the image ------------------ """

# Load the the data and use the coordinates in array X and the greyscale values 
# Z to plot the point cloud image Z.

# load the .npz file
coffe = cv2.imread('Mandetory\Supplementary data\coffee.png', 0)
point_cloud = np.load('Mandetory\Supplementary data\point_cloud.npz')
laplacian = np.load('Mandetory\Supplementary data\point_cloud_Laplacian.npz')

# extract the data  
X = point_cloud['X']
Z = point_cloud['Z']

L = laplacian['L']

plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
plt.ylim(400, 0)
plt.xticks([]), plt.yticks([])
plt.show() 


""" ------------------ 2. histogram for image ------------------ """

hist, bins = np.histogram(Z, bins=100)

plt.bar(bins[:-1], hist, width=0.008, color='black')
plt.title('Histogram of the point cloud intensity values')
plt.show()

"""  ------------------ 3. Graph Fourier mode ------------------ """

eigenvalues, eigenvectors = np.linalg.eig(L)


# ordering the eigenvalues 
eigenvalues = np.sort(eigenvalues)

# plotting the eigenvalues
plt.plot(eigenvalues, color = 'black')
plt.title('Eigenvalues of the Laplacian')
plt.show()
