import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy as sc

plt.style.use('ggplot')

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')
JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')

gray_JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png',0)
blue, green, red = cv2.split(JUP1)


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
    magnitude_spectrum = np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title("Magnitude Spectrum", x=-0.1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

#histogram(JUP1)
magnitude_spectrum(red)
#magnitude_spectrum(gray_JUP2)

""" Task 1c """


#JUP1_merge = cv2.merge([blue, green, red])

""" sp_filtered = sc.ndimage.filters.median_filter(cv2.merge([blue, green, red]), size = 3)
cv2.imshow("Restored Jupiter1", sp_filtered) """
def notch_filter(shape, d0, u_k, v_k):
    M, N = shape
    H = np.zeros((M, N))
    
    for u in range(0, M):
        for v in range(0, N):
            D_k = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
            D_mk = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
            if D_k <= d0 or D_mk <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0
    return H

f = np.fft.fft2(red) 
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))

H1 = notch_filter(red.shape, 4, 8, 8)
H2 = notch_filter(red.shape, 4, 8, -8)
red = fshift * H1
red = np.fft.ifftshift(red)
red = np.fft.ifft2(red)
red = np.abs(red)

red = np.array(red, dtype=np.uint8)

plt.imshow(magnitude_spectrum*H2, cmap = 'gray')
plt.show()

plt.imshow(magnitude_spectrum*H1, cmap = 'gray')
plt.show()

plt.imshow(red, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

#sp_and_p_filtered = sc.ndimage.filters.median_filter(cv2.merge([blue, green, red]), size = 3)

cv2.imshow("Restored Jupiter1", cv2.merge([blue, green, red]))
        
cv2.waitKey(0)  