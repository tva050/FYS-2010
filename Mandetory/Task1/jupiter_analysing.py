from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio

plt.style.use('ggplot')

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')
JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')
gray_JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png', 0)
gray_JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png', 0)
""" Task 1b """
def analyse_image():
    def histogram(img):
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color = col)
            plt.xlim([0,256])
            plt.title("Histogram of Jupiter1")
        plt.show()
        
        histogram, bin_edges = np.histogram(JUP2 , bins=256, range=(0, 255))
        plt.bar(bin_edges[0:-1], histogram, width = 0.7)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.title("Histogram of Jupiter2")
        plt.show()
        
    def magnitude_spectrum(img): 
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    
    histogram(JUP1)
    magnitude_spectrum(gray_JUP1)
    magnitude_spectrum(gray_JUP2)
    
    

analyse_image()

