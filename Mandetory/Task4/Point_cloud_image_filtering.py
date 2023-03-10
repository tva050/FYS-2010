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
Laplacian = np.load('Mandetory\Supplementary data\point_cloud_Laplacian.npz')

# extract the data  
X = point_cloud['X']
Z = point_cloud['Z']

L = Laplacian['L']

def plot_point_cloud_image():
    plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.show() 


""" ------------------ 2. histogram for image ------------------ """

hist, bins = np.histogram(Z, bins=100)

def plot_histogram():
    plt.bar(bins[:-1], hist, width=0.008, color='black')
    plt.title('Histogram of the point cloud intensity values')
    plt.show()

"""  ------------------ 3. Graph Fourier mode (eig) ------------------ """

eigenvalues, eigenvectors = np.linalg.eigh(L) # eigenvalues and eigenvectors of the Laplacian


idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]


# checking that they are all positive if not, there is an error 
if np.all(eigenvalues < 0) == False:
    print('Error: the eigenvalues are not all positive')
    print('The smallest eigenvalue is: ', eigenvalues[0])
    
    


# plotting the eigenvalues
def plot_eigenvalues():
    plt.plot(eigenvalues, color = 'black')
    plt.title('Eigenvalues of the Laplacian')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.show()


""" ------------------ 4. Fourier mode ------------------ """

eigenvector1 = eigenvectors[:,1]  # first non-zero eigenvalue
eigenvector2 = eigenvectors[:,2]
eigenvector3 = eigenvectors[:,3]
eigenvector4 = eigenvectors[:,4]
eigenvector5  = eigenvectors[:,5]
eigenvector6 = eigenvectors[:,6]

def plot_modes():
    plt.subplot(2,3,1)
    plt.scatter(X[:,1], X[:,0], c=eigenvector1, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('mode 1')
    plt.subplot(2,3,2)
    plt.scatter(X[:,1], X[:,0], c=eigenvector2, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('mode 2')
    plt.subplot(2,3,3)
    plt.scatter(X[:,1], X[:,0], c=eigenvector3, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('mode 3')
    plt.subplot(2,3,4)
    plt.scatter(X[:,1], X[:,0], c=eigenvector4, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('mode 4')
    plt.subplot(2,3,5)
    plt.scatter(X[:,1], X[:,0], c=eigenvector5, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('mode 5')
    plt.subplot(2,3,6)
    plt.scatter(X[:,1], X[:,0], c=eigenvector6, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('mode 6')
    plt.show() 
    
eigenvector2300 = eigenvectors[:,2300] # 2300th eigenvalue

def plot_eigenvector2300():
    plt.scatter(X[:,1], X[:,0], c=eigenvector2300, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.show()


""" ------------------ 5. Grap Fourier Transform ------------------ """

graph_fourier = np.inner(eigenvectors.T, Z) # graph Fourier transform of the image Z (Source: https://en.wikipedia.org/wiki/Graph_Fourier_transform)

# plotting the graph Fourier transform 
def plot_graph_fourier():
    plt.plot(eigenvalues, graph_fourier, color = 'black')
    plt.title('Graph Fourier transform')
    plt.xlabel('Laplacian eigenvalue')
    plt.ylabel('Fourier coefficient')
    plt.show()
        
    
""" ------------------ 6. Filtering lowpass ------------------ """
# Filter the graph Fourier transform with a low-pass filter
c_lowpass = 1 # cut-off frequency

low_pass = np.zeros(len(graph_fourier)) # low pass filter with length of graph Fourier transform
low_pass = np.where(eigenvalues < c_lowpass, graph_fourier, 0) # low pass filter
def plot_low_pass():
    plt.plot(eigenvalues, low_pass, color = 'black')
    plt.title('Graph Fourier transform')
    plt.xlabel('Laplacian eigenvalue')
    plt.ylabel('Fourier coefficient')
    plt.show()
    
    inverse_graph_fourier = np.dot(eigenvectors, low_pass) # inverse graph Fourier transform
    
    plt.subplot(1,2,1)
    plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('Original image')
    plt.subplot(1,2,2)
    plt.scatter(X[:,1], X[:,0], c=inverse_graph_fourier, cmap='gray') # inverse graph Fourier transform
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('Low pass filtered')
    plt.show()

"""  ------------------ 7. Filtering highpass ------------------ """
# Filter the graph Fourier transform with a high-pass filter
c_highpass = 1 # cut-off frequency

high_pass = np.zeros(len(graph_fourier))
high_pass = np.where(eigenvalues > c_highpass, graph_fourier, 0) # high pass filter, where the eigenvalues are larger than the cut-off frequency

def plot_high_pass():
    plt.plot(eigenvalues, high_pass, color = 'black')
    plt.title('Graph Fourier transform')
    plt.xlabel('Laplacian eigenvalue')
    plt.ylabel('Fourier coefficient')
    plt.show()

    inverse_graph_fourier = np.dot(eigenvectors, high_pass) # inverse graph Fourier transform
    
    plt.subplot(1,2,1)
    plt.scatter(X[:,1], X[:,0], c=Z, cmap='gray')
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('Original image')
    plt.subplot(1,2,2)
    plt.scatter(X[:,1], X[:,0], c=inverse_graph_fourier, cmap='gray') # inverse graph Fourier transform
    plt.ylim(400, 0)
    plt.xticks([]), plt.yticks([])
    plt.title('High pass filtered')
    plt.show()
    
if __name__ == '__main__':
    #plot_point_cloud_image()
    #plot_histogram()
    #plot_eigenvalues()
    #plot_modes()
    #plot_eigenvector2300()
    #plot_graph_fourier()
    #plot_low_pass()
    #plot_high_pass()