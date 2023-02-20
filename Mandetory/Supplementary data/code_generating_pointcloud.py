# How the graph was constructed and the Laplacian
import numpy as np
import skimage
import sklearn.neighbors

# load the original image (and save it black&white for the exam)
original = skimage.data.coffee()

img = skimage.color.rgb2gray(original)
img = skimage.img_as_ubyte(img)
skimage.io.imsave('coffee.png',img)
img = skimage.io.imread('coffee.png')
img = skimage.img_as_float(img)

# Let us construct the point cloud
# First step, select randomly a subset of the image pixels
N = 3000
seed=5
np.random.seed(seed)
x_r = np.random.randint(0,img.shape[0],N)
y_r = np.random.randint(0,img.shape[1],N)
val = [(c[0],c[1],img[c[0],c[1]]) for c in zip(x_r,y_r)]
x,y,z = zip(*val)
# x,y contains the coordinates and z is the intensity of each pixel
# We make then numpy arrays
X = np.zeros((N,2))
X[:,0] = x
X[:,1] = y
Z = np.array(z)
np.savez_compressed('point_cloud.npz',X=X,Z=Z)

# Creating the graph by connecting the nearest neighbors
# We will compute the euclidean distance between pixels to find the nearest neighbors.
# We use the scikit-learn function that compute all the distances and find the nearest neighbors
# It returns the adjacency matrix
number_nearest_neighbors = 4
A = sklearn.neighbors.kneighbors_graph(X, number_nearest_neighbors)
# The returned adjacency matrix is not symmetric.
# We make it symmetric
A = np.array(A.todense())
As = A + A.T
As[As>1] = 1
# We compute the degree matrix (diagonal matrix)
D = np.diag(np.sum(As, axis=0))
# The Laplacian L is:
L = D - As
np.savez_compressed('point_cloud_Laplacian.npz',L=L)
