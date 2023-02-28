import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndi
import scipy as sc

plt.style.use('ggplot')

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')
JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')

gray_JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png',0)
blue, green, red = cv2.split(JUP1)


""" --------------- Task 1b ---------------"""
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
#magnitude_spectrum(red)
#magnitude_spectrum(gray_JUP2)


""" --------------- Task 1c ---------------"""

orginal = cv2.merge((blue, green, red))
# __Notch filter__
# Source: https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python
def notch_filter(shape, d0, u_k, v_k): 
    M, N = shape
    H = np.zeros((M, N))
    
    for u in range(0, M):
        for v in range(0, N):
            D_k = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
            D_mk = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
            horizontal1 = u - M/2 + u_k 
            vertical1 =v - N/2 + v_k
            horizontal2 = u - M/2 - u_k 
            vertical2 = v - N/2 - v_k
        
            if D_k <= d0 or D_mk <= d0 or horizontal1 == 0 or vertical1 == 0 or horizontal2 == 0 or vertical2 == 0:
                H[u, v] = 0.0
            else: 
                H[u, v] = 1.0
    return H
f = np.fft.fft2(red) 
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))

H1 = notch_filter(red.shape, 3, 7, 7)

red = fshift * H1
red = np.fft.ifftshift(red)
red = np.fft.ifft2(red)
red = np.abs(red)
red = np.uint8(red)

notch_filtered = cv2.merge((blue, green, red))

# __Median filter__
median_filter = ndi.median_filter(cv2.merge((blue, green, red)), size=3)

# __Contrast Harmonic Mean filter__
np.seterr(invalid="ignore") # To ignore the warning of division by zero
# Contrast Harmonic Mean filter, 
# written with help from https://stackabuse.com/introduction-to-image-processing-in-python-with-opencv/
def CHM_filter(img, Q):
    img = img.astype(np.float64)
    numirator = img**(Q+1)
    denumirator = (img)**(Q)
    kernel = np.full(shape = 3, fill_value= 1.0, dtype = np.float64)
    result = cv2.filter2D(numirator, -1, kernel) / cv2.filter2D(denumirator, -1, kernel)
    return result

CHM_filtered = CHM_filter(median_filter, -2.)
CHM_filtered = np.uint8(CHM_filtered)

# __Plotting__
def plot_notch_filtered_J1img_mag():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(orginal, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Orginal Jupiter1")
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum*H1, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Eliminated bursts")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter1")
    plt.show()

def plot_median_CHM_filtered_J1img():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter1")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Median filtered")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(CHM_filtered, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("CHM filtered")
    plt.show()
    
def restored_JUP1_image():
    plt.imshow(cv2.cvtColor(CHM_filtered, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Restored Jupiter1")
    plt.show()
    
# plot_notch_filtered_J1img_mag()
# plot_median_CHM_filtered_J1img()
restored_JUP1_image()


""" --------------- Task 1d ---------------"""

cv2.waitKey(0)  