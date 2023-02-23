from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy as sc
import scipy.io as sio

plt.style.use('ggplot')

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')
JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')

gray_JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png', 0) # Jupiter2 in grayscale
blue, green, red = cv2.split(JUP1)  # Jupiter1 in RGB


""" Task 1b """
def histogram(img):
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
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
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title("Magnitude Spectrum", x=-0.1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def restore_jupiter1():
    # Starting by filter out the salt-and-pepper noise using madian filter
    JUp1 = np.array(Image.open('Mandetory\Supplementary data\Jupiter1.png'))
    jup1_medfiltered = sc.ndimage.median_filter(JUp1, size=3)
    plt.imshow(jup1_medfiltered)
    plt.xticks([])
    plt.yticks([])
    plt.title("Median filter")
    plt.show()
    
    def notch_filter(shape, d0=9, u_k=0, v_k=0):
        P, Q = shape
        # initialize the filter with zeros
        H = np.zeros((P, Q))

        # Travers through the filter and calculate the value of each pixel
        for u in range(0, P):
            for v in range(0, Q):
                D_k = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
                D_mk = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)
                if D_k <= d0 or D_mk <= d0:
                    H[u, v] = 0.0
                else:
                    H[u, v] = 1.0
        return H

    #gray_JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png',0)
    gray_JUP1 = red

    f = np.fft.fft2(gray_JUP1)
    fshift = np.fft.fftshift(f)
    phase_spectrumR = np.angle(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    img_shape =  red.shape 

    H1 = notch_filter(img_shape, 4, 5, 10)

    NotchFilter = H1 
    NotchRejectCenter = fshift * NotchFilter 
    NotchReject = np.fft.ifftshift(NotchRejectCenter)
    inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result

    Result = np.abs(inverse_NotchReject)

    plt.imshow(Result, "gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filter")
    plt.show()        

    plt.imshow(magnitude_spectrum*NotchFilter, "gray") 
    plt.title("Notch Reject Filter")
    plt.xticks([])
    plt.yticks([])
    plt.show()
#histogram(JUP1) # Jupiter1 in RGB
#magnitude_spectrum(red) # red channel of Jupiter1
#magnitude_spectrum(gray_JUP2) # Jupiter2 in grayscale
restore_jupiter1()
    


